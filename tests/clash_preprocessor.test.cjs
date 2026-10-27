const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');

function parser(address = '100.64.0.10') {
  const template = fs.readFileSync(path.join(__dirname,'../deploy/chat/clash-preprocessor.example.yml'),'utf8');
  const code = template.split('    code: |\n')[1].replace(/^      /gm,'').replace('YOUR_ALIYUN_TAILSCALE_IP',address);
  const context = {module: {exports: {}}};
  vm.runInNewContext(code,context);
  // JSON is valid YAML. Keep dependency-free tests focused on the transformation.
  return input => JSON.parse(context.module.exports.parse(JSON.stringify(input), {yaml: {parse: JSON.parse, stringify: JSON.stringify}}));
}

const original = {
  'mixed-port':7890,mode:'rule',dns:{enable:false},
  proxies:[{name:'fixture-node',type:'socks5',server:'127.0.0.1',port:19000}],
  'proxy-groups':[{name:'Existing',type:'select',proxies:['DIRECT','fixture-node']}],
  rules:['IP-CIDR,100.64.0.0/10,DIRECT,no-resolve','MATCH,Existing']
};

test('only adds the host-specific Tailscale group and first rule', () => {
  const result = parser()(original);
  assert.deepEqual(result['proxy-groups'].slice(1),original['proxy-groups']);
  assert.deepEqual(result.rules.slice(1),original.rules);
  assert.equal(result.rules[0],'IP-CIDR,100.64.0.10/32,MemoryDraft-Tailscale,no-resolve');
  assert.deepEqual(result['proxy-groups'][0],{name:'MemoryDraft-Tailscale',type:'select',proxies:['DIRECT'],'interface-name':'Tailscale'});
  for (const key of ['mixed-port','mode','dns','proxies']) assert.deepEqual(result[key],original[key]);
});

test('repeated preprocessing is idempotent and replaces only its own group', () => {
  const apply = parser(), once = apply(original);
  assert.deepEqual(apply(once),once);
  once['proxy-groups'][0]['interface-name']='obsolete';
  assert.equal(apply(once)['proxy-groups'][0]['interface-name'],'Tailscale');
  assert.equal(apply(once)['proxy-groups'].length,2);
});

test('invalid input and an unconfigured address fail explicitly', () => {
  assert.throws(()=>parser('YOUR_ALIYUN_TAILSCALE_IP')(original),/Set the chat/);
  assert.throws(()=>parser()(null),/configuration object/);
  assert.throws(()=>parser()({...original,rules:{}}),/arrays/);
});

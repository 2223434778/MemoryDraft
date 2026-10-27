async (page) => {
  const report={runs:[],mobile:[]}, errors=[];
  page.on('pageerror', error=>errors.push(error.message));
  try {
  await page.emulateMedia({reducedMotion:'no-preference'});
  await page.reload();
  await page.setViewportSize({width:1440,height:1000});
  await page.locator('#model-status').filter({hasText:'已就绪'}).waitFor();
  report.brand={title:await page.title(),oldName:await page.locator('body').innerText().then(s=>s.includes('MemoryDraft')),
    favicon:(await page.request.get('http://127.0.0.1:8840/static/favicon.svg')).status()};
  if(report.brand.oldName||!report.brand.title.startsWith('树流')||report.brand.favicon!==200) throw new Error('Brand mismatch');
  report.glass=await page.locator('#chat-form').evaluate(n=>({blur:getComputedStyle(n).backdropFilter,background:getComputedStyle(n).backgroundColor}));
  await page.evaluate(()=>{
    const original=window.fetch; window.__runs=[];
    window.fetch=async (...args)=>{
      const response=await original(...args);
      if(String(args[0]).endsWith('/messages')&&args[1]?.method==='POST') {
        const record={url:args[0],request:JSON.parse(args[1].body)}; window.__runs.push(record);
        response.clone().text().then(raw=>{
          record.speculation=raw.split('\n\n').filter(b=>b.startsWith('event: speculation')).length;
          const block=raw.split('\n\n').find(b=>b.startsWith('event: complete'));
          if(block) record.final=JSON.parse(block.split('data: ')[1]);
          const error=raw.split('\n\n').find(b=>b.startsWith('event: error'));
          if(error) record.error=JSON.parse(error.split('data: ')[1]);
        }).catch(()=>{});
      }
      return response;
    };
  });
  async function details(value) {
    if(await page.locator('#process-panel').evaluate(n=>n.hidden)===value) await page.locator('#process-toggle').click();
  }
  async function choose(id,value) {
    await page.locator(`#${id}-button`).click();
    await page.locator(`#${id}-menu [data-value="${value}"]`).click();
  }
  for(const [inline,detail] of [[false,false],[false,true],[true,false],[true,true]]) {
    await page.locator('#new-chat').click();
    await page.locator('#visualize').setChecked(inline); await details(detail);
    await choose('method-select',inline&&!detail?'DFlash':'BudgetTree-ASD');
    await page.locator('#prompt').fill('用中文详细解释数据库事务的原理、隔离级别和实际用途，给出多个具体例子。');
    await page.locator('#send').click();
    await page.waitForFunction(()=>Number(document.getElementById('stat-round').textContent)>1);
    const state=await page.evaluate(()=>({inline:document.querySelectorAll('.inline-speculation').length,
      details:!document.getElementById('process-panel').hidden,checkboxEnabled:!document.getElementById('visualize').disabled,
      generating:!document.getElementById('stop').hidden}));
    if(state.inline!==Number(inline)||state.details!==detail||!state.checkboxEnabled||!state.generating) throw new Error(JSON.stringify({inline,detail,state}));
    if(inline&&detail) {
      await page.locator('#visualize').uncheck();
      if(await page.locator('.inline-speculation,.confirmed-token').count()||await page.locator('#process-panel').evaluate(n=>n.hidden)) throw new Error('Inline off affected details');
      await page.locator('#visualize').check(); await details(false);
      if(!await page.locator('.inline-speculation').count()||!await page.locator('#visualize').isChecked()) throw new Error('Details close affected inline');
      await details(true);
      report.liveTogglesIndependent=true;
    }
    if(detail) {
      await page.locator('#process-panel').evaluate(n=>Promise.all(n.getAnimations().map(a=>a.finished.catch(()=>{}))));
      await page.screenshot({path:`output/playwright/shuliu-${inline?'both':'details-only'}.png`});
    }
    await page.locator('#stop').waitFor({state:'hidden',timeout:60000});
    await page.waitForFunction(()=>window.__runs.at(-1)?.final||window.__runs.at(-1)?.error);
    const record=await page.evaluate(()=>window.__runs.at(-1));
    if(record.error) throw new Error(JSON.stringify(record.error));
    const cid=record.url.split('/')[3];
    const saved=await (await page.request.get('http://127.0.0.1:8840/api/conversations/'+cid)).json();
    const result={inline,detail,events:record.speculation,requestVisualize:record.request.visualize,
      savedEqualsComplete:saved.messages.at(-1).content===record.final.text,
      finalMarkers:await page.locator('.inline-speculation,.confirmed-token').count(),
      detailsUnchanged:await page.locator('#process-panel').evaluate(n=>!n.hidden)===detail,
      assistantName:await page.locator('.message.assistant .message-name').last().innerText()};
    if(!result.events||!result.requestVisualize||!result.savedEqualsComplete||result.finalMarkers||!result.detailsUnchanged||result.assistantName!=='树流') throw new Error(JSON.stringify(result));
    report.runs.push(result);
    if(!inline&&!detail) {
      await details(true);
      if(await page.locator('#visualize').isChecked()||!Number(await page.locator('#stat-round').innerText())) throw new Error('Closed views lost trace');
      await details(false); report.reopenRetainsActualEvents=true;
    }
  }
  await details(false);
  await page.locator('#visualize').uncheck(); await page.locator('#visualize').focus();
  await page.keyboard.press('Space');
  report.checkbox=await page.locator('.check-shell').evaluate(n=>({width:parseFloat(getComputedStyle(n).width),
    height:parseFloat(getComputedStyle(n).height),radius:getComputedStyle(n).borderRadius,
    animation:getComputedStyle(n).animationName,outline:getComputedStyle(n).outlineStyle}));
  if(!await page.locator('#visualize').isChecked()||report.checkbox.width<24) throw new Error(JSON.stringify({checkbox:report.checkbox,checked:await page.locator('#visualize').isChecked()}));
  await page.locator('#model-select-button').click();
  await page.locator('#model-select-menu').evaluate(n=>Promise.all(n.getAnimations().map(a=>a.finished.catch(()=>{}))));
  report.portal=await page.locator('#model-select-menu').evaluate(n=>n.parentElement===document.body);
  await page.screenshot({path:'output/playwright/shuliu-menu.png'}); await page.keyboard.press('Escape');
  await choose('model-select','qwen3.8-27b-q8');
  await page.locator('#method-select-button').click();
  report.disabledModes=await page.locator('#method-select-menu button:disabled').count();
  await page.keyboard.press('Escape'); await choose('model-select','qwen3-8b');
  await choose('method-select','BudgetTree-ASD');
  for(const width of [390,320]) {
    await page.setViewportSize({width,height:844});
    await page.locator('#model-select-button').click();
    await page.locator('#model-select-menu').evaluate(n=>Promise.all(n.getAnimations().map(a=>a.finished.catch(()=>{}))));
    const bounds=await page.evaluate(()=>{
      const menu=document.getElementById('model-select-menu').getBoundingClientRect(),button=document.getElementById('model-select-button').getBoundingClientRect(),header=document.getElementById('process-toggle').getBoundingClientRect();
      return {width:innerWidth,document:document.documentElement.scrollWidth,left:menu.left,right:menu.right,bottom:menu.bottom,anchorTop:button.top,headerRight:header.right};
    });
    if(bounds.document!==width||bounds.left<0||bounds.right>width||bounds.bottom>bounds.anchorTop||bounds.headerRight>width) throw new Error(JSON.stringify(bounds));
    report.mobile.push(bounds);
    await page.screenshot({path:`output/playwright/shuliu-mobile-${width}.png`}); await page.keyboard.press('Escape');
  }
  await page.emulateMedia({reducedMotion:'reduce'});
  await page.locator('#visualize').uncheck(); await page.locator('#visualize').check();
  report.reducedMotion=await page.locator('.check-shell').evaluate(n=>({animation:getComputedStyle(n).animationName,transition:getComputedStyle(n).transitionDuration}));
  if(report.reducedMotion.animation!=='none'||report.reducedMotion.transition!=='0s') throw new Error('Reduced motion not respected');
  report.pageErrors=errors;
  if(errors.length) throw new Error(JSON.stringify(errors));
  return report;
  } finally { await page.evaluate(value=>{window.__designReport=value;},report); }
}

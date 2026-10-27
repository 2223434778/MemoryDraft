async (page) => {
  const report={gpuRuns:[],layouts:[]}, errors=[];
  page.on('pageerror', error=>errors.push(error.message));
  await page.emulateMedia({reducedMotion:'no-preference'});
  await page.setViewportSize({width:1440,height:1000});
  await page.reload();
  await page.locator('#model-status').filter({hasText:'已就绪'}).waitFor();
  await page.evaluate(()=>{
    const original=window.fetch; window.__glassRuns=[];
    window.fetch=async (...args)=>{
      const response=await original(...args);
      if(String(args[0]).endsWith('/messages')&&args[1]?.method==='POST') {
        const record={request:JSON.parse(args[1].body)}; window.__glassRuns.push(record);
        response.clone().text().then(raw=>{
          record.speculationEvents=raw.split('\n\n').filter(b=>b.startsWith('event: speculation')).length;
          record.complete=raw.includes('event: complete\n');
        }).catch(()=>{});
      }
      return response;
    };
  });
  const button=page.locator('#process-toggle'), bounds=await button.boundingBox();
  await page.mouse.move(bounds.x+10,bounds.y+7);
  await page.waitForFunction(()=>document.getElementById('process-toggle').classList.contains('glass-tracking'));
  const read=()=>button.evaluate(n=>({x:n.style.getPropertyValue('--glass-x'),angle:n.style.getPropertyValue('--glass-angle'),highlight:getComputedStyle(n,'::before').backgroundImage}));
  const first=await read();
  await page.mouse.move(bounds.x+bounds.width-10,bounds.y+bounds.height-7);
  await page.waitForFunction(x=>document.getElementById('process-toggle').style.getPropertyValue('--glass-x')!==x,first.x);
  const second=await read();
  if(first.x===second.x||first.angle===second.angle) throw new Error('Reflection did not follow pointer');
  report.reflection={first,second};
  await page.screenshot({path:'output/playwright/liquid-history-welcome.png'});
  await page.emulateMedia({reducedMotion:'reduce'});
  await page.mouse.move(bounds.x+20,bounds.y+10);
  if(await button.evaluate(n=>n.classList.contains('glass-tracking'))) throw new Error('Reduced-motion reflection active');
  report.reducedMotion=true;
  await page.emulateMedia({reducedMotion:'no-preference'});
  async function saved() { return page.evaluate(async()=>({...(await (await api('/api/conversations/'+selected)).json())})); }
  async function send(text,count) {
    await page.locator('#prompt').fill(text); await page.locator('#send').click();
    await page.waitForFunction(count=>document.querySelectorAll('.message.assistant').length===count&&!document.getElementById('send').hidden&&!document.querySelector('.generating')&&[...document.querySelectorAll('.message-action')].every(n=>!n.disabled),count,{timeout:180000});
    const data=await saved();
    if(data.messages.at(-1).status!=='complete') throw new Error('Real GPU response incomplete');
    return data;
  }
  const originalQuestion='请只用一句中文解释什么是数据库索引。';
  await send(originalQuestion,1);
  const original=await send('请给一个简短的索引例子。',2);
  const sourceId=original.id;
  await page.locator('.message.user').first().getByRole('button',{name:'编辑提问',exact:true}).click();
  await page.getByRole('textbox',{name:'修改这条提问'}).fill('取消的内容');
  await page.getByRole('textbox',{name:'修改这条提问'}).press('Escape');
  if(await page.locator('.message-editor').count()||JSON.stringify(await saved())!==JSON.stringify(original)) throw new Error('Cancel changed history');
  report.cancelPreservesOriginal=true;
  await page.locator('.message.user').nth(1).getByRole('button',{name:'编辑提问',exact:true}).click();
  const editedQuestion='只用一句话解释数据库索引为什么能加速查询。';
  await page.getByRole('textbox',{name:'修改这条提问'}).fill(editedQuestion);
  await page.screenshot({path:'output/playwright/liquid-history-editor-desktop.png'});
  await page.getByRole('button',{name:'保存并重新生成',exact:true}).click();
  await page.waitForFunction(()=>document.querySelectorAll('.message.assistant').length===2&&!document.getElementById('send').hidden&&!document.querySelector('.generating')&&[...document.querySelectorAll('.message-action')].every(n=>!n.disabled),null,{timeout:180000});
  const edited=await saved();
  if(edited.id===sourceId||edited.messages[2].content!==editedQuestion||edited.messages.length!==4) throw new Error('Question edit did not create expected branch');
  const sourceAfter=await page.evaluate(async id=>(await api('/api/conversations/'+id)).json(),sourceId);
  if(JSON.stringify(sourceAfter)!==JSON.stringify(original)) throw new Error('Original conversation changed');
  report.questionEdit={originalPreserved:true,branchMessages:edited.messages.length,regenerated:true};
  await page.locator('.message.assistant').first().getByRole('button',{name:'编辑回答',exact:true}).click();
  const editedAnswer='这是手工编辑的上下文：索引像目录，能帮助数据库更快定位记录。';
  await page.getByRole('textbox',{name:'修改这段回答'}).fill(editedAnswer);
  await page.getByRole('button',{name:'保存并继续',exact:true}).click();
  await page.waitForFunction(()=>!document.querySelector('.message-editor')&&!document.getElementById('send').hidden&&document.querySelectorAll('.message').length===2);
  const answerBranch=await saved();
  if(answerBranch.messages[1].content!==editedAnswer||!answerBranch.messages[1].metadata.history_edit||answerBranch.messages[1].metadata.speculation) throw new Error('Edited answer retained stale trace or wrong content');
  report.answerEdit={branchMessages:2,traceCleared:true};
  await send('请用一句话继续说明索引的代价。',2);
  await page.locator('.message.assistant').first().getByRole('button',{name:'从这里继续',exact:true}).click();
  await page.waitForFunction(()=>!document.getElementById('send').hidden&&document.querySelectorAll('.message').length===2);
  const rewound=await saved();
  if(rewound.messages[1].content!==editedAnswer) throw new Error('Answer rewind used future context');
  await page.locator('.message.user').first().getByRole('button',{name:'回到这里重新提问',exact:true}).click();
  await page.waitForFunction(()=>!document.getElementById('send').hidden&&document.querySelectorAll('.message').length===0);
  if(await page.locator('#prompt').inputValue()!==originalQuestion) throw new Error('Question rewind lost draft');
  report.rewind={answerMessages:2,firstQuestionMessages:0,promptRestored:true};
  for(const width of [390,320]) {
    await page.setViewportSize({width,height:844});
    await page.locator('#menu').click();
    await page.getByRole('button',{name:original.title,exact:true}).click();
    await page.locator('.message.user').first().getByRole('button',{name:'编辑提问',exact:true}).click();
    const layout=await page.evaluate(()=>({width:innerWidth,documentWidth:document.documentElement.scrollWidth,editorWidth:document.querySelector('.message-editor').getBoundingClientRect().width,textareaWidth:document.querySelector('.message-editor textarea').getBoundingClientRect().width}));
    if(layout.documentWidth>width+1||layout.textareaWidth>width) throw new Error('Mobile overflow '+JSON.stringify(layout));
    report.layouts.push(layout);
    await page.screenshot({path:`output/playwright/liquid-history-mobile-${width}.png`});
    await page.getByRole('button',{name:'取消',exact:true}).click();
  }
  await page.setViewportSize({width:1440,height:1000});
  await page.locator('#process-toggle').click();
  await page.screenshot({path:'output/playwright/liquid-history-final-desktop.png'});
  report.gpuRuns=await page.evaluate(()=>window.__glassRuns.map(r=>({method:r.request.method,visualize:r.request.visualize,speculationEvents:r.speculationEvents,complete:r.complete})));
  report.pageErrors=errors;
  if(errors.length||report.gpuRuns.length!==4||report.gpuRuns.some(r=>!r.complete||!r.speculationEvents)) throw new Error('GPU/UI check failed '+JSON.stringify(report));
  return report;
}

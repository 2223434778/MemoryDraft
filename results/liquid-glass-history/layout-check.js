async (page) => {
  const report={layouts:[]};
  async function settle() {
    await page.evaluate(async()=>{await Promise.all(document.getAnimations().filter(a=>a.effect.getTiming().iterations!==Infinity).map(a=>a.finished.catch(()=>{})))});
  }
  await page.reload();
  await page.locator('#model-status').filter({hasText:'已就绪'}).waitFor();
  await page.getByRole('button',{name:'请只用一句中文解释什么是数据库索引。',exact:true}).click();
  await page.locator('.message.user').nth(1).getByRole('button',{name:'编辑提问',exact:true}).click();
  await settle();
  await page.screenshot({path:'output/playwright/liquid-history-editor-desktop.png'});
  await page.getByRole('button',{name:'取消',exact:true}).click();
  for(const width of [390,320]) {
    await page.setViewportSize({width,height:844});
    await page.locator('.message.user').first().getByRole('button',{name:'编辑提问',exact:true}).click();
    await settle();
    const layout=await page.evaluate(()=>{
      const e=document.querySelector('.message-editor').getBoundingClientRect(),s=document.getElementById('scroll-area').getBoundingClientRect();
      return {width:innerWidth,documentWidth:document.documentElement.scrollWidth,editorTop:e.top,editorBottom:e.bottom,scrollTop:s.top,scrollBottom:s.bottom,saveVisible:document.querySelector('.editor-save').getBoundingClientRect().bottom<=s.bottom+1};
    });
    if(layout.documentWidth!==width||!layout.saveVisible||layout.editorTop<layout.scrollTop-1) throw new Error('Editor accessibility '+JSON.stringify(layout));
    report.layouts.push(layout);
    await page.screenshot({path:`output/playwright/liquid-history-mobile-${width}.png`});
    await page.getByRole('button',{name:'取消',exact:true}).click();
  }
  await page.setViewportSize({width:1440,height:1000});
  await page.locator('#process-toggle').click(); await settle();
  await page.screenshot({path:'output/playwright/liquid-history-final-desktop.png'});
  report.desktop=await page.evaluate(()=>({width:innerWidth,documentWidth:document.documentElement.scrollWidth,sidebarX:document.querySelector('.sidebar').getBoundingClientRect().x,processOpacity:getComputedStyle(document.querySelector('.process-panel')).opacity}));
  if(report.desktop.documentWidth!==1440||Math.abs(report.desktop.sidebarX)>1||report.desktop.processOpacity!=='1') throw new Error('Unsettled desktop layout');
  return report;
}

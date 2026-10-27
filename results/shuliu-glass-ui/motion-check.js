async (page) => {
  await page.emulateMedia({reducedMotion:'no-preference'});
  await page.locator('#visualize').uncheck(); await page.locator('#visualize').focus();
  await page.keyboard.press('Space');
  const motion=await page.locator('.check-shell').evaluate(n=>({checked:document.getElementById('visualize').checked,
    animation:getComputedStyle(n).animationName,duration:getComputedStyle(n).animationDuration,
    pathTransition:getComputedStyle(n.querySelector('path')).transitionDuration,
    width:getComputedStyle(n).width,height:getComputedStyle(n).height,
    hitWidth:document.getElementById('visualize').getBoundingClientRect().width}));
  if(!motion.checked||motion.animation!=='check-settle'||motion.duration!=='0.28s') throw new Error(JSON.stringify(motion));
  await page.setViewportSize({width:1440,height:1000});
  await page.locator('#process-toggle').click();
  await page.locator('#process-panel').evaluate(n=>Promise.all(n.getAnimations().map(a=>a.finished.catch(()=>{}))));
  await page.screenshot({path:'output/playwright/shuliu-final-desktop.jpg',type:'jpeg',quality:90});
  await page.locator('#process-close').click();
  await page.setViewportSize({width:390,height:844});
  await page.locator('#model-select-button').click();
  await page.locator('#model-select-menu').evaluate(n=>Promise.all(n.getAnimations().map(a=>a.finished.catch(()=>{}))));
  await page.screenshot({path:'output/playwright/shuliu-final-mobile.jpg',type:'jpeg',quality:90});
  await page.keyboard.press('Escape');
  return motion;
}

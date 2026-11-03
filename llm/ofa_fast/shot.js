// usage: node shot.js <scratch-dir> <port> <out.png> [wait-seconds]   headless-Chrome DevTools screenshot of the open-flapping-airplanes dashboard
// (the persistent /stream makes --screenshot/--virtual-time-budget hang, so we drive Chrome over the DevTools websocket)
const {spawn}=require('child_process');const fs=require('fs');
const [S,PORT,OUT,WAIT]=[process.argv[2],process.argv[3]||'8099',process.argv[4]||'dash.png',+(process.argv[5]||9)];
const ch=spawn('/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',['--headless=new','--disable-gpu','--remote-debugging-port=9334','--user-data-dir='+S+'/chrshot','--window-size=1400,900','about:blank'],{stdio:'ignore'});
const sleep=ms=>new Promise(r=>setTimeout(r,ms));
(async()=>{
  let t;for(let i=0;i<40;i++){try{t=await (await fetch('http://localhost:9334/json')).json();if(t.length)break}catch(e){}await sleep(250)}
  const ws=new WebSocket(t.find(x=>x.type==='page').webSocketDebuggerUrl);await new Promise(r=>ws.onopen=r);
  let id=0;const pend={};ws.onmessage=m=>{const d=JSON.parse(m.data);if(d.id&&pend[d.id])pend[d.id](d.result)};
  const send=(method,params={})=>new Promise(r=>{const i=++id;pend[i]=r;ws.send(JSON.stringify({id:i,method,params}))});
  await send('Page.enable');await send('Page.navigate',{url:'http://localhost:'+PORT+'/'});await sleep(WAIT*1000);
  const r=await send('Runtime.evaluate',{expression:'JSON.stringify({got:gotFrames,fps:shownFps,loss:$("loss").textContent,step:$("step").textContent,tag:[$("ttag").textContent,$("itag").textContent,$("gtag").textContent],itok:$("itok").textContent,txt:document.getElementById("out").textContent.slice(0,120)})',returnByValue:true});
  console.log(r.result.value);
  const sh=await send('Page.captureScreenshot',{format:'png'});fs.writeFileSync(OUT,Buffer.from(sh.data,'base64'));
  ch.kill();process.exit(0)})();

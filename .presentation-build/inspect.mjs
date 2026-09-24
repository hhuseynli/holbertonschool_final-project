import fs from 'node:fs/promises';
import {FileBlob, PresentationFile} from '@oai/artifact-tool';
const ref='C:/Users/BAKU/.codex/plugins/cache/openai-curated-remote/openai-templates/0.1.1/skills/artifact-template-simple-dark-mode/assets/reference.pptx';
const p=await PresentationFile.importPptx(await FileBlob.load(ref));
await fs.writeFile('.presentation-build/template-inspect.json', (await p.inspect({kind:'slide,textbox,image,chart,table',maxChars:200000})).ndjson);
await fs.writeFile('.presentation-build/template-proto.json',JSON.stringify(p.toProto()));
for(let i=0;i<p.slides.items.length;i++){
 const s=p.slides.items[i];
 await fs.writeFile(`.presentation-build/ref-${i+1}.png`,new Uint8Array(await (await s.export({format:'png',scale:0.65})).arrayBuffer()));
 console.log(i+1, s.shapes.items.map(x=>({id:x.id,t:x.text.toString().slice(0,65),pos:x.position})), 'images',s.images.items.length,'charts',s.charts.items.length);
}

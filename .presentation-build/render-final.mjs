import fs from 'node:fs/promises';
import {FileBlob,PresentationFile} from '@oai/artifact-tool';
const p=await PresentationFile.importPptx(await FileBlob.load('output/BakuML_Project_Presentation.pptx'));
for(let i=0;i<p.slides.items.length;i++){
 await fs.writeFile(`.presentation-build/final-${i+1}.png`,new Uint8Array(await (await p.slides.items[i].export({format:'png',scale:1})).arrayBuffer()));
}
console.log('Rendered',p.slides.items.length,'final slides');

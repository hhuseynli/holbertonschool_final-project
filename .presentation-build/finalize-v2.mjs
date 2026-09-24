import fs from 'node:fs/promises';
import path from 'node:path';
import {pathToFileURL} from 'node:url';
import {FileBlob,PresentationFile} from '@oai/artifact-tool';
process.env.RUNTIME_NODE_MODULES='C:/Users/BAKU/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules';
const root=process.cwd(),b=path.join(root,'.presentation-build');
const skill='C:/Users/BAKU/.codex/plugins/cache/openai-primary-runtime/presentations/26.909.12148/skills/presentations';
const {finalizePresentation}=await import(pathToFileURL(skill+'/container_tools/artifact_tool_utils.mjs'));
const final=path.join(root,'output/BakuML_Future_City_Presentation.pptx');
console.log(await finalizePresentation({workspaceDir:root,candidatePath:b+'/candidate-v2.pptx',finalPath:final,pythonExecutable:'C:/Users/BAKU/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe',integrityValidatorPath:skill+'/container_tools/inspect_presentation_package_integrity.py',layoutValidatorPath:skill+'/container_tools/inspect_presentation_layout_geometry.py',layoutArgs:['--expected-slide-size-emu','12192000,6858000','--validate-bullet-geometry','--validate-heading-fit'],requiredNativeChartOwnerSlides:[4,5,8,10],requiredNativeTableOwnerSlides:[],materializeLiteralChartWorkbooks:true,fontPolicy:{basis:'design',families:['Arial']},verifyArtifactToolImport:true,receiptPath:b+'/validation-v2.json'}));
const p=await PresentationFile.importPptx(await FileBlob.load(final));
for(let i=0;i<p.slides.items.length;i++){const v=await p.slides.items[i].export({format:'png',scale:1});await fs.writeFile(b+`/final-v2-${i+1}.png`,new Uint8Array(await v.arrayBuffer()));}

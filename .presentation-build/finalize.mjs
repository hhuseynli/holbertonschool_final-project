import fs from 'node:fs/promises';
import path from 'node:path';
import crypto from 'node:crypto';
import {pathToFileURL} from 'node:url';
const workspaceDir=process.cwd();
const skill='C:/Users/BAKU/.codex/plugins/cache/openai-primary-runtime/presentations/26.909.61513/skills/presentations';
const ref='C:/Users/BAKU/.codex/plugins/cache/openai-curated-remote/openai-templates/0.1.1/skills/artifact-template-simple-dark-mode/assets/reference.pptx';
const {finalizePresentation}=await import(pathToFileURL(path.join(skill,'container_tools/artifact_tool_utils.mjs')).href);
console.log(await finalizePresentation({workspaceDir,candidatePath:path.join(workspaceDir,'.presentation-build/candidate.pptx'),finalPath:path.join(workspaceDir,'output/BakuML_Project_Presentation.pptx'),pythonExecutable:'C:/Users/BAKU/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe',integrityValidatorPath:path.join(skill,'container_tools/inspect_presentation_package_integrity.py'),layoutValidatorPath:path.join(skill,'container_tools/inspect_presentation_layout_geometry.py'),layoutArgs:['--expected-slide-size-emu','12192000,6858000','--validate-bullet-geometry','--validate-heading-fit','--require-native-table-slide','18'],requiredNativeChartOwnerSlides:[6,10,12],requiredNativeTableOwnerSlides:[18],materializeLiteralChartWorkbooks:true,fontPolicy:{basis:'reference',families:['Helvetica Neue','Helvetica Neue Medium'],referencePath:ref,referenceSha256:crypto.createHash('sha256').update(await fs.readFile(ref)).digest('hex')},verifyArtifactToolImport:true,receiptPath:path.join(workspaceDir,'.presentation-build/validation.json')}));


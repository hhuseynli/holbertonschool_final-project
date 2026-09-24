import fs from 'node:fs/promises';
import path from 'node:path';
import crypto from 'node:crypto';
import {FileBlob,Presentation,PresentationFile} from '@oai/artifact-tool';
import {pathToFileURL} from 'node:url';
const ROOT=process.cwd(), BUILD=path.join(ROOT,'.presentation-build');
const SKILL='C:/Users/BAKU/.codex/plugins/cache/openai-primary-runtime/presentations/26.909.61513/skills/presentations';
const REF='C:/Users/BAKU/.codex/plugins/cache/openai-curated-remote/openai-templates/0.1.1/skills/artifact-template-simple-dark-mode/assets/reference.pptx';
const {finalizePresentation,applyPresentationChartFont}=await import(pathToFileURL(path.join(SKILL,'container_tools/artifact_tool_utils.mjs')).href);
const imported=await PresentationFile.importPptx(await FileBlob.load(REF));
const proto=imported.toProto();
const source=structuredClone(proto.slides);
const layouts=[1,2,13,5,17,22,13,6,5,22,19,21,6,17,13,26,5,14,13];
proto.slides=layouts.map((n,i)=>({...structuredClone(source[n-1]),id:`baku-slide-${i+1}`,index:i}));
// Clone the retained template layouts and edit their existing content in place.
function el(n,id){return proto.slides[n-1].elements.find(x=>x.id===String(id));}
function txt(n,id,lines,sizes){
 const e=el(n,id); if(!e)throw Error(`missing ${n}/${id}`);
 const old=e.paragraphs?.length?e.paragraphs:[{runs:[{}]}];
 e.paragraphs=(Array.isArray(lines)?lines:[lines]).map((t,i)=>{
  const p=structuredClone(old[Math.min(i,old.length-1)]), r=structuredClone(p.runs?.[0]||{});
  r.text=t;delete r.fieldType;r.id='';r.textStyle={...r.textStyle};
  if(sizes)r.textStyle.fontSize=100*(Array.isArray(sizes)?sizes[i]||sizes.at(-1):sizes);
  p.runs=[r];p.inlineNodes=[];p.id='';return p;
 });
}
function title(n,text){const e=proto.slides[n-1].elements.find(x=>x.placeholderType==='title');txt(n,e.id,text,30);}
function body(n,id,head,text,size=18){txt(n,id,[head,text],[24,size]);}
function foot(n,text){const e=proto.slides[n-1].elements.find(x=>x.placeholderType==='ftr');if(e){txt(n,e.id,text,10);e.bbox={xEmu:393700,yEmu:6370000,widthEmu:10400000,heightEmu:241300};}}
for(let i=1;i<=layouts.length;i++){
 const num=proto.slides[i-1].elements.find(x=>x.placeholderType==='sldNum');if(num)txt(i,num.id,String(i),10);
}
txt(1,4,['Beyond Price','Prediction'],54);
txt(1,5,['Baku & Absheron real estate','Holberton School  /  Group 3'],18);
txt(1,6,'BakuML',20);
txt(2,4,['Where will Baku’s','market move next?'],60);
txt(2,6,'The research question',18);txt(2,8,'Asking prices, place and time',18);
title(3,'BakuML combines a marketplace with spatial research');
body(3,10,'Property discovery','Filter listings by district, budget, rooms and building type.');
body(3,11,'Listing analytics','Compare asking prices with cell estimates and a composite score.');
body(3,4,'Interactive geography','Explore map layers, metro stations and cell-level histories.');
body(3,5,'Research engine','Evaluate spatial models and infrastructure scenarios offline.');
foot(3,'Current interface: Flask, Jinja, Leaflet and Plotly');
title(4,'Two data paths, two levels of evidence');
body(4,7,'Synthetic research panel','Controlled monthly listings with known duplicates, price gradients and a simulated station opening.\n\nSupports temporal validation and recovery of planted effects.',20);
body(4,8,'Real-listing snapshot','The bina.az GraphQL client feeds a separate cross-sectional model.\n\nA snapshot supports price comparisons. Long-term growth outputs remain heuristic.',20);
foot(4,'The figures in this presentation are documented synthetic benchmarks');
title(5,'The research pipeline');
txt(5,9,'01  Data',14);txt(5,13,'02  Geography',14);txt(5,14,'03  Models',14);
body(5,15,'Clean the listings','Normalize fields and remove broker reposts using image hashes and text similarity.',17);
body(5,8,'Build the panel','Assign spatial cells, aggregate monthly prices and compute past-only features.',17);
body(5,19,'Publish the outputs','Evaluate models, estimate intervals and save artifacts for the application.',17);
title(6,'Adaptive cells retain more usable history');
body(6,6,'Geography changes the dataset','Coordinate-only KD-tree splits adapt to listing density. The documented default balances coverage and spatial transfer.',18);
txt(6,38,'5',50);txt(6,39,['Median months','H3 resolution 8'],14);
txt(6,41,'26',50);txt(6,42,['Median months','KD-tree ~140/cell'],14);
foot(6,'Documented synthetic MAUP study. Chart: percentage of listings retained');
title(7,'Features connect price to place and time');
body(7,10,'Urban structure','Distances to the city centre and configured polycentric nodes.');
body(7,11,'Transit access','Distance to open metro stations and the number within 2 km.');
body(7,4,'Local market history','Own-price lags, momentum and previous-month neighbour prices.');
body(7,5,'Building mix','Previous-month supply composition and redevelopment-zone flags.');
foot(7,'Current distances use haversine geometry. Network travel distance is future work');
title(8,'A model stack with distinct responsibilities');
body(8,534,'XGBoost','Estimate cell-month asking prices. TreeSHAP explains the fitted model’s price drivers.',18);
body(8,3,'Conformal quantiles','Fit q10, q50 and q90, then calibrate the interval on later months.',18);
body(8,2,'Graph forecasting','STGCN uses adjacency and history for 12/24-month rollouts. Ridge provides a fallback.',18);
foot(8,'The reported MAE and R² benchmarks measure XGBoost, not long-horizon STGCN accuracy');
title(9,'Validation separates time from geography');
body(9,7,'Temporal walk-forward','Train on past months and evaluate later months.\n\nCompare each split with persistence: next month equals the last observed month.',20);
body(9,8,'Spatial blocked folds','Hold out contiguous regions across months. Exclude neighbour-price features in this evaluation.\n\nA canary test perturbs future prices and checks earlier features.',20);
foot(9,'These safeguards describe the synthetic research pipeline');
title(10,'XGBoost beats persistence on the synthetic panel');
body(10,6,'Mean absolute error','Lower is better, in AZN/m².\n\n25 temporal splits and 5 spatial folds in the documented full run.',18);
txt(10,38,'19%',46);txt(10,39,['Lower temporal MAE','213 versus 264'],14);
txt(10,41,'17%',46);txt(10,42,['Lower spatial MAE','209 versus 253'],14);
foot(10,'README full-run benchmarks. Rounded improvement: (naive MAE − model MAE) / naive MAE');
title(11,'A simulated metro opening tests causal recovery');
body(11,14,'Spatial difference-in-differences','Compare listings within 1 km with controls 2–6 km away. Cell and month effects isolate the simulated treatment.',18);
txt(11,13,'0.069',48);txt(11,10,['Estimated average','effect, log units'],16);
txt(11,4,'0.075',48);txt(11,12,['Approximate planted','ramp-average effect'],16);
txt(11,5,'0.088',48);txt(11,11,['Post-ramp estimate','versus 0.080 planted'],16);
foot(11,'Documented synthetic recovery. This is not an estimate of an observed real-world metro effect');
title(12,'Uncertainty remains an open challenge');
txt(12,43,'Held-out coverage falls below the 80% target. A trending market challenges exchangeability.',18);
txt(12,38,'71.4%',43);txt(12,39,['Observed coverage','on unseen months'],14);
txt(12,41,'8.6 pp',43);txt(12,42,['Gap below the','80% target'],14);
foot(12,'Synthetic CQR holdout. Multi-month interval widths use a √horizon heuristic');
title(13,'Scenarios explore explicit assumptions');
body(13,534,'Baseline','Continue the fitted forecast without an additional policy adjustment.',19);
body(13,3,'Polycentric','Double the estimated historical node-gravity appreciation slope.',19);
body(13,2,'Transit','Apply the estimated metro effect near a hypothetical next station.',19);
foot(13,'Research scenarios are conditional simulations, not promised investment returns');
title(14,'The product demonstration');
txt(14,9,'01  Browse',14);txt(14,13,'02  Inspect',14);txt(14,14,'03  Explore',14);
body(14,15,'Find a property','Apply district and budget filters, then sort the results.',18);
body(14,8,'Read the context','Open the detail page to inspect price position, metro proximity and score.',18);
body(14,19,'Compare the map','Switch metrics and horizons, then inspect a cell’s history and estimates.',18);
foot(14,'A populated demo requires listing and model artifacts');
title(15,'The next milestone is credible real-data validation');
body(15,10,'Collect longitudinal data','Build a recurring listing history before evaluating future growth.');
body(15,11,'Rebuild validation boundaries','Compute neighbour features within folds and reserve calibration data.');
body(15,4,'Unify the application geometry','Make listing enrichment support the research pipeline’s adaptive cells.');
body(15,5,'Calibrate longer horizons','Backtest 12/24-month forecasts and measure coverage separately.');
txt(16,4,['BakuML','Questions?'],60);
txt(16,6,'Group 3',18);
txt(16,5,['Hüseyn Hüseynli   ·   Nihad Süleymanov   ·   Eldəniz Arifzadə','Zeynəb Mirzəzadə   ·   Vüqar Dünyamalıyev'],17);
el(16,5).bbox.widthEmu=11000000;
title(17,'Appendix: current implementation boundaries');
body(17,7,'Real-data evaluation','Neighbour prices use the full snapshot before spatial folds. This can contaminate held-out regions.\n\nInterval residuals come from the fitted training data, not an independent calibration set.',18);
body(17,8,'Application integration','Listing enrichment hard-codes H3, while the research default uses KD-tree cells.\n\nThe real pipeline writes baseline outputs only. The map offers all configured scenarios.',18);
foot(17,'Source review of scripts/run_real_pipeline.py and app/enrichment.py');
title(18,'Appendix: geographic sensitivity');
txt(18,14,'Documented synthetic study. Skill = 1 − model MAE / persistence MAE.',17);
foot(18,'Positive skill beats persistence on spatially held-out regions. Default: KD-tree ~140/cell');
title(19,'Appendix: evidence and reproducibility');
body(19,10,'Recovery-based tests','Duplicate detection and DiD tests compare estimates with planted truth.');
body(19,11,'Leakage checks','Tests cover future-price perturbations, geographic partitions and time splits.');
body(19,4,'Documented benchmarks','Numerical results come from README.md. This checkout contains no saved run artifacts.');
body(19,5,'Current verification scope','Source and test inspection only. Dependencies needed to rerun the suite are absent.');
// Replace template example charts/tables with the project's evidence, keeping their layout frames.
for(const n of [6,10,12])proto.slides[n-1].elements=proto.slides[n-1].elements.filter(e=>!e.chartReference);
proto.slides[17].elements=proto.slides[17].elements.filter(e=>e.id!=='534');
const p=Presentation.load(proto);
const cover=p.slides.items[0].images.items[0];
const frame=cover.frame;
cover.replace({blob:new Uint8Array(await fs.readFile(path.join(BUILD,'baku-concept.png'))),contentType:'image/png',alt:'AI-generated conceptual illustration of Baku waterfront and spatial analysis',fit:'cover'});cover.frame=frame;
// Repair a pre-existing timeline line that extends outside the template canvas.
for(const n of [5,14]){const s=p.slides.items[n-1].shapes.items.find(x=>x.id==='2');s.position={left:35.46,top:354.2,width:1203.2,height:0.032};}
const FONT='Helvetica Neue';
function chart(n,categories,series,max,unit){
 const s=p.slides.items[n-1];
 const c=s.charts.add('bar',{position:{left:66,top:177,width:526,height:421},categories,series,barOptions:{direction:'column',grouping:'clustered',gapWidth:110},hasLegend:series.length>1,legend:{position:'bottom',textStyle:{typeface:FONT,fontSize:16,fill:'#FFFFFF'}},chartFill:'#000000',plotAreaFill:'#000000',chartLine:{fill:'none',width:0},xAxis:{textStyle:{typeface:FONT,fontSize:17,fill:'#FFFFFF'},majorGridlines:null},yAxis:{min:0,max,majorUnit:max===100?20:100,title:{text:unit,textStyle:{typeface:FONT,fontSize:15,fill:'#FFFFFF'}},textStyle:{typeface:FONT,fontSize:15,fill:'#BBBBBB'},majorGridlines:{fill:'#333333',width:0.6}},dataLabels:{showValue:true,position:'outEnd',textStyle:{typeface:FONT,fontSize:20,fill:'#FFFFFF'}}});
 applyPresentationChartFont(c,{fontFamily:FONT});
}
chart(6,['H3 res 8','KD ~140'],[{name:'Listings retained',values:[62.6,81.6],fill:'#51C4E6'}],100,'Listings retained (%)');
chart(10,['Temporal','Spatial'],[{name:'Persistence',values:[264,253],fill:'#DBE6EB'},{name:'XGBoost',values:[213,209],fill:'#51C4E6'}],350,'MAE (AZN/m²)');
chart(12,['Target','Observed'],[{name:'Coverage',values:[80,71.4],fill:'#51C4E6'}],100,'Coverage (%)');
const values=[['Spatial unit','Listings kept','Median months','Transfer skill'],['H3 res 7','93.7%','33','−0.174'],['H3 res 8','62.6%','5','+0.139'],['H3 res 9','11.3%','2','−0.339'],['KD-tree ~70','46.1%','9','+0.189'],['KD-tree ~140','81.6%','26','+0.174'],['KD-tree ~280','98.5%','42','+0.159'],['Market k=120','98.0%','42','−3.601'],['Market k=240','94.8%','14','−0.608']];
const t=p.slides.items[17].tables.add({rows:9,columns:4,left:42,top:230,width:1195,height:410,columnWidths:[405,263,263,264],values});
t.borders.assign({fill:'#555555',width:0.5});
for(let r=0;r<9;r++)for(let c=0;c<4;c++){const cell=t.getCell(r,c);cell.fill=r===5?'#123D4A':'#000000';cell.text.style={typeface:FONT,fontSize:r===0?19:21,color:'#FFFFFF',bold:r===0};}
const notes=[
`About 25 seconds. Introduce BakuML as a graduation project studying asking-price patterns across Baku and Absheron. The project combines a research pipeline with a property-browsing application. The question is whether spatial structure and infrastructure can help explain where the market changes. Source: README.md; app/templates/base.html. Cover artwork is an AI-generated conceptual illustration, not a factual geographic map or a photograph.`,
`About 35 seconds. A listing contains a current asking price. Our research question adds geography and time: where might prices change as access and urban structure change? Explain that asking prices are observable in advertisements, while executed transaction prices are outside the dataset. Urban expansion is an ambition in the README, not a separately validated land-change output in the current implementation. Source: README.md; bakuml/pipeline.py.`,
`About 40 seconds. Describe the two halves of the project. Flask serves the Buy, listing detail and Map pages. JavaScript calls listing, cell and map APIs. The analytics layer enriches listings using precomputed artifacts. The research pipeline creates those artifacts offline. This is an implemented interface, but the current checkout has empty data/artifacts folders, so avoid claiming a populated live deployment. Sources: app/flask_app.py, app/enrichment.py, app/templates/, app/static/js/.`,
`About 45 seconds. Make the evidence boundary explicit early. The synthetic generator plants price gradients, duplicates and a station treatment. That lets us test whether the system recovers known quantities. The real pipeline consumes a short bina.az snapshot and estimates listing-level price/m². Its 12/24-month growth columns use node gravity multiplied by fixed constants, not a learned temporal forecast. No real-data benchmark artifacts exist in this checkout. Sources: bakuml/data/synthetic.py; bakuml/data/scraping/client.py; scripts/run_real_pipeline.py.`,
`About 40 seconds. Data normalization creates consistent fields. Dedup uses spatial/property blocking plus image hash or TF-IDF similarity, then chooses a canonical listing. Tessellation creates cell-month observations. Feature engineering uses historical data, followed by validation, XGBoost explanations, conformal quantiles, graph forecasting and spatial DiD. Parquet/JSON outputs decouple offline computation from the app. Sources: bakuml/data/dedup.py; bakuml/pipeline.py; bakuml/data/schema.py.`,
`About 50 seconds. Explain the modifiable areal unit problem: changing boundaries changes which listings share an estimate. Fixed H3 cells can be thin where listing density is low. The default KD-tree uses only coordinates to balance counts. The documented MAUP study retains 81.6% versus 62.6% for H3 res 8 and gives 26 versus 5 median months. The ~70 setting has slightly higher skill but less retained data. These are source-reported synthetic measurements, not a new run. Sources: README.md spatial-architecture table; scripts/maup_study.py; bakuml/spatial/tessellation.py.`,
`About 40 seconds. Distinguish static location from historical price features. Master-plan features use configured approximate node coordinates and redevelopment areas. They do not yet represent a complete GIS ingestion of the official plan. Metro features use the station opening timeline. Price and supply lags look backward. Current distances are straight-line haversine distances. Sources: bakuml/config.py; bakuml/features/masterplan.py; bakuml/features/infrastructure.py; bakuml/features/build.py.`,
`About 45 seconds. XGBoost predicts monthly cell median asking price. TreeSHAP ranks contributions to this fitted model. CQR trains quantile models and adjusts bounds using later calibration months. STGCN learns temporal and graph structure and rolls forward autoregressively, with a spatial-lag Ridge fallback when torch is unavailable. Do not attach the XGBoost MAE/R² to the STGCN's 12/24-month forecasts. Sources: bakuml/models/baseline.py; conformal.py; stgcn.py; bakuml/pipeline.py.`,
`About 50 seconds. Randomly mixing nearby listings or future months creates overly easy evaluation. Temporal walk-forward trains strictly before its evaluation window and compares with persistence. Spatial CV removes entire contiguous groups from model training and drops neighbour target encoding in the research pipeline. Own-cell historical price features remain available, so interpret spatial results as model transfer with local history, not a geography with no data at all. The canary test triples future prices and asserts earlier feature rows are unchanged. Sources: bakuml/models/baseline.py; bakuml/pipeline.py; tests/test_features.py.`,
`About 50 seconds. These README benchmarks use the full synthetic run. Temporal MAE is 213 against 264 AZN/m², approximately 19.3% lower. Spatial MAE is 209 against 253, approximately 17.4% lower. README reports model R² 0.81 in both evaluations. The full documented panel has 6,799 usable cell-months and 3,953 feature/training rows. This is evidence that the method works on the controlled data-generating process, not evidence of live-market accuracy. Source: README.md Results on the offline dataset.`,
`About 45 seconds. Explain the difference between proximity correlation and treatment effect. The synthetic design compares treated listings within one kilometre with controls two to six kilometres away, excluding the buffer. The model includes cell and month fixed effects and controls, with cluster-robust errors. Average post-opening ATT is 0.069 log units against approximately 0.075 planted ramp average. Post-ramp event-study mean is 0.088 against 0.080 planted. These are distinct estimands. Event-study pre-trends are documented near zero. This recovery experiment does not establish a causal effect in real Baku. Sources: README.md; bakuml/causal/did.py; tests/test_did.py.`,
`About 45 seconds. CQR uses a train/calibrate/holdout time split: six calibration months and a final three-month holdout. Report the observed 71.4% coverage honestly against an 80% target, a gap of 8.6 percentage points. Exchangeability is difficult with trends and drift. The research pipeline reuses relative one-month widths and scales them by square root of horizon for longer forecasts. That heuristic does not establish calibrated 12/24-month coverage. Sources: README.md; bakuml/pipeline.py; bakuml/models/conformal.py.`,
`About 40 seconds. The research pipeline composes scenarios from estimated quantities, not planted truth. Baseline keeps the forecast unchanged. Polycentric doubles the estimated historical gravity slope. Transit places a hypothetical B-05 one inter-station step beyond B-04 and applies the estimated ATT under an assumed opening/ramp schedule. These are sensitivity exercises. The real-data pipeline currently emits only baseline rows, so demonstrate all three only with compatible research artifacts. Source: bakuml/pipeline.py; scripts/run_real_pipeline.py.`,
`About 45 seconds, or 90 seconds if a populated app is available. Browse by district and budget, open one listing, explain the analytics and switch to the map. Describe the score as a weighted heuristic. Do not describe it as a validated return probability. Before presenting, use compatible H3 artifacts for the current listing enricher, because it does not resolve KD-tree cell IDs. No populated dataset ships in this checkout. Sources: app/flask_app.py; app/enrichment.py; app/templates/detail.html; app/static/js/map.js.`,
`About 45 seconds. Prioritize the real-data evidence gap. Collect longitudinal snapshots, perform dedup, create neighbour features within the training fold and separate calibration from final evaluation. Unify listing-to-cell mapping and artifact geometry. Then backtest each long horizon and scenario honestly. OSM network distances and satellite embeddings are later extensions. Source-review findings: scripts/run_real_pipeline.py; app/enrichment.py; README.md future work.`,
`About 20 seconds. Close on the contribution: an adaptable spatial research framework and a property interface that makes its outputs accessible. Invite questions about geography, leakage, causal design or real-data readiness. Team names reproduced from README.md. No individual contribution assignments were available in the repository.`,
`Backup slide. In the real pipeline, cell_medians and nbr_price are computed on fm before cv_fold is used. As a result, test-region labels can influence features used across folds. The final model trains on all rows and derives residual quantiles on those same rows, so reported interval coverage is in-sample. The real pipeline does not invoke the research dedup stage and creates gravity-based appreciation plus a normal-CDF score. The listing enricher directly uses h3.latlng_to_cell, which does not match KD IDs. The map offers config.SCENARIOS despite the real pipeline producing baseline only. These are source-review findings, not runtime-verified failures. Sources: scripts/run_real_pipeline.py; app/enrichment.py; app/flask_app.py.`,
`Backup slide. All eight rows reproduce the README MAUP table. Positive transfer skill indicates lower MAE than persistence under blocked spatial evaluation. KD ~140 is the default compromise, not the maximum-skill row. Market regions improve homogeneity but perform poorly under this transfer setup. Do not generalize this ranking to every dataset. Sources: README.md; scripts/maup_study.py.`,
`Backup slide. Tests in the repository cover normalization, dedup recovery, features, validation, spatial tessellation, baseline models, DiD, forecasting, visualization and the pipeline. The code review inspected the tests but did not execute pytest, because the available runtime lacks pytest and the project ML/geospatial dependencies. Local artifacts/ and data/ contain no datasets or benchmark outputs. Reproduce documented numbers with the project dependencies and the full pipeline, preserving separate artifact directories for research and real data. README and DESIGN still refer to Streamlit/Scrapy, whereas the current implementation uses Flask and a GraphQL client. Sources: tests/; requirements.txt; README.md; DESIGN.md; Makefile.`
];
for(let i=0;i<p.slides.items.length;i++)p.slides.items[i].speakerNotes.textFrame.setText(notes[i]);
await fs.writeFile(path.join(BUILD,'speaker-notes.txt'),notes.map((x,i)=>`${i+1}. ${x}`).join('\n\n'));
const candidate=path.join(BUILD,'candidate.pptx');
await (await PresentationFile.exportPptx(p)).save(candidate);
console.log('exported candidate');
for(let i=0;i<p.slides.items.length;i++){
 const blob=await p.slides.items[i].export({format:'png',scale:1});
 await fs.writeFile(path.join(BUILD,`slide-${i+1}.png`),new Uint8Array(await blob.arrayBuffer()));
}
const finalPath=path.join(ROOT,'output','BakuML_Project_Presentation.pptx');
const receipt=await finalizePresentation({workspaceDir:ROOT,candidatePath:candidate,finalPath,pythonExecutable:'C:/Users/BAKU/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe',integrityValidatorPath:path.join(SKILL,'container_tools/inspect_presentation_package_integrity.py'),layoutValidatorPath:path.join(SKILL,'container_tools/inspect_presentation_layout_geometry.py'),layoutArgs:['--expected-slide-size-emu','12192000,6858000','--validate-bullet-geometry','--validate-heading-fit'],requiredNativeChartOwnerSlides:[6,10,12],requiredNativeTableOwnerSlides:[18],fontPolicy:{basis:'reference',families:['Helvetica Neue'],referencePath:REF,referenceSha256:crypto.createHash('sha256').update(await fs.readFile(REF)).digest('hex')},verifyArtifactToolImport:true,receiptPath:path.join(BUILD,'validation.json')});
console.log(JSON.stringify(receipt));

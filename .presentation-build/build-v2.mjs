import fs from 'node:fs/promises';
import path from 'node:path';
import {pathToFileURL} from 'node:url';
import {Presentation,PresentationFile} from '@oai/artifact-tool';
const ROOT=process.cwd(),B=path.join(ROOT,'.presentation-build');
const SKILL='C:/Users/BAKU/.codex/plugins/cache/openai-primary-runtime/presentations/26.909.12148/skills/presentations';
const {applyPresentationChartFont}=await import(pathToFileURL(SKILL+'/container_tools/artifact_tool_utils.mjs'));
const e=JSON.parse(await fs.readFile(B+'/evidence.json','utf8'));
const m=e.kd_metrics, wf=m.walk_forward,sp=m.spatial_cv;
const p=Presentation.create({slideSize:{width:1280,height:720}});
const C={paper:'#F5F2EB',ink:'#123C43',teal:'#138C85',muted:'#607879',coral:'#DE775A',pale:'#D6E9E5',white:'#FFFFFF',dark:'#102F37'};
const FONT='Arial';
function text(s,t,x,y,w,h,size=28,color=C.ink,bold=false){let z=s.shapes.add({geometry:'textbox',position:{left:x,top:y,width:w,height:h},fill:'none',line:{fill:'none',width:0}});z.text=t;z.text.style={typeface:FONT,fontSize:size,color,bold,autoFit:'none'};return z;}
function slide(title,sub='',bg=C.paper){let s=p.slides.add();s.background.fill=bg;const fg=bg===C.dark?C.white:C.ink;text(s,title,60,38,1160,100,42,fg,true);if(sub)text(s,sub,62,135,1145,52,20,bg===C.dark?C.pale:C.muted);return s;}
function note(s,t){s.speakerNotes.textFrame.setText(t+'\n\nEvidence: fresh full synthetic pipeline run, seed 20243, 18 September 2026. Artifacts in .presentation-build/research-kdtree unless stated otherwise. No live-market forecast accuracy is established.');}
function foot(s,t,dark=false){text(s,t,62,668,1156,33,14,dark?C.pale:C.muted);}
async function img(s,name,x,y,w,h,fit='contain'){s.images.add({blob:new Uint8Array(await fs.readFile(B+'/'+name)),contentType:'image/png',fit,position:{left:x,top:y,width:w,height:h}});}
function chart(s,type,categories,series,{x=60,y=208,w=770,h=412,max=undefined,min=0,unit='',horizontal=false,dark=false,labels=true}={}){let bg=dark?C.dark:C.paper,fg=dark?C.white:C.ink;let c=s.charts.add(type,{position:{left:x,top:y,width:w,height:h},categories,series,barOptions:{direction:horizontal?'bar':'column',grouping:'clustered',gapWidth:95},hasLegend:series.length>1,legend:{position:'bottom',textStyle:{typeface:FONT,fontSize:17,fill:fg}},chartFill:bg,plotAreaFill:bg,chartLine:{fill:'none',width:0},xAxis:{textStyle:{typeface:FONT,fontSize:17,fill:fg},majorGridlines:null},yAxis:{min,max,title:{text:unit,textStyle:{typeface:FONT,fontSize:16,fill:fg}},textStyle:{typeface:FONT,fontSize:16,fill:fg},majorGridlines:{fill:dark?'#35545A':'#D2DDDA',width:.6}},dataLabels:{showValue:labels,position:'outEnd',textStyle:{typeface:FONT,fontSize:18,fill:fg}}});applyPresentationChartFont(c,{fontFamily:FONT});return c;}
const round=x=>Math.round(x), pct=x=>(100*x).toFixed(1);

// 1: A visual opening, with the research objective stated in one sentence.
{
let s=p.slides.add();s.background.fill=C.paper;
await img(s,'baku-concept.png',655,0,625,720,'cover');
text(s,'BakuML',60,55,560,50,28,C.teal,true);
text(s,'Where might\nBaku grow next?',58,170,605,210,58,C.ink,true);
text(s,'Mapping urban potential through\n12- and 24-month real estate\nprice scenarios.',62,427,560,132,27);
text(s,'Holberton School · Group 3',62,638,560,38,19,C.muted);
note(s,'The project connects asking prices, geography, infrastructure and time to explore future urban market potential. This is the purpose, not a claim of a validated urban-expansion model. Cover is an AI-generated conceptual Baku illustration, not a geographic map or photograph. Sources: README.md; bakuml/pipeline.py; bakuml/config.py.');
}
// 2: Actual H3 output, with a large map and clearly identified evidence type.
{
let s=slide('A city of different price patterns','Baku & Absheron · latest available synthetic median asking prices, AZN/m²');
await img(s,'price-map.png',60,180,1160,480,'contain');
foot(s,'H3 resolution 8 · cells with usable data · cell observation dates can differ');
note(s,'Map generated from the project’s hex_layer_geojson helper and the fresh H3 artifacts. Each cell uses its latest available observed synthetic median asking price; this is not a simultaneous real-market snapshot. Colour limits use the 5th and 95th percentiles, with outliers saturated. Basemap attribution remains visible. Sources: research-h3/panel.parquet, predictions.parquet, cell_geometry.geojson; bakuml/viz.py.');
}
{
let s=slide('44 months turn listings into a research panel','January 2023 – August 2026 · controlled synthetic experiment');
text(s,'37,312',65,232,570,105,82,C.teal,true);text(s,'generated listing records',68,344,535,55,27);
text(s,'2,117',710,232,500,105,82,C.coral,true);text(s,'duplicate reposts removed',712,344,495,55,27);
text(s,'256 spatial cells',68,489,535,55,34,C.ink,true);text(s,'6,799 observed cell-months',712,489,505,55,34,C.ink,true);
foot(s,'KD-tree research run · known synthetic truth lets us test the method');
note(s,'Dataset counts are fresh run metrics, not README estimates. 35,195 deduplicated records remain; the minimum cell-month count is three. The feature matrix contains 3,953 rows after history availability constraints. Source: research-kdtree/metrics.json dataset.');
}
{
let s=slide('The model improves on “last month again”','Mean absolute error in AZN/m² · lower is better');
chart(s,'bar',['Future months','Held-out regions'],[{name:'Persistence',values:[round(wf.naive.mae),round(sp.naive.mae)],fill:'#AAC2BF'},{name:'XGBoost',values:[round(wf.aggregate.mae),round(sp.aggregate.mae)],fill:C.teal}],{max:350,unit:'MAE (AZN/m²)'});
text(s,pct(1-wf.aggregate.mae/wf.naive.mae)+'%',898,254,300,105,70,C.teal,true);text(s,'lower temporal error',899,358,295,66,25);
text(s,pct(1-sp.aggregate.mae/sp.naive.mae)+'% lower',899,468,305,60,34,C.ink,true);text(s,'across held-out regions',899,535,305,75,23);
foot(s,'Fresh synthetic KD-tree run · these tests measure monthly price estimates');
note(s,`Temporal: ${JSON.stringify(wf.aggregate)} versus persistence ${JSON.stringify(wf.naive)} across ${wf.n_splits} splits. Spatial: ${JSON.stringify(sp.aggregate)} versus ${JSON.stringify(sp.naive)} across ${sp.n_splits} folds. Percent improvement uses unrounded MAE. Spatial evaluation omits neighbour encoding but retains own-cell price history, so this is not prediction in locations with no history. These metrics do not validate the 12/24-month forecast. Sources: metrics.json; bakuml/models/baseline.py.`);
}
{
let s=slide('Location is the strongest fitted price signal','Mean absolute TreeSHAP contribution · AZN/m²');
let keys=Object.keys(e.shap).slice(0,5),names={'dist_centre_km':'City-centre distance','nbr_price_prev_month':'Neighbour prices','month_ix':'Time trend','dist_metro_km':'Metro distance','lag_own_1m':'Last own-cell price'};
chart(s,'bar',keys.map(k=>names[k]||k),[{name:'Mean absolute SHAP',values:keys.map(k=>round(e.shap[k])),fill:C.teal}],{w:1120,max:360,horizontal:true,unit:'Mean |SHAP| (AZN/m²)'});
foot(s,'Global importance explains this fitted synthetic model; it does not establish causality');
note(s,'The five largest mean absolute TreeSHAP contributions come directly from shap_summary.json. City-centre distance dominates, followed by neighbour prices and time. This is consistent with the synthetic generator’s planted spatial gradient, not independent proof of a real-world mechanism. Source: research-kdtree/shap_summary.json; bakuml/models/baseline.py; bakuml/data/synthetic.py.');
}
{
let s=slide('Can we recover a simulated metro effect?','Spatial difference-in-differences · a test against planted truth',C.dark);
text(s,e.did.att_log.toFixed(3),66,228,560,112,84,C.white,true);text(s,'estimated average effect\n(log-price units)',69,357,530,95,28,C.pale);
text(s,'≈0.075',720,228,480,112,84,C.coral,true);text(s,'planted ramp-average effect\n(log-price units)',723,357,480,95,28,C.pale);
text(s,'The experiment recovers a positive effect.',68,541,1130,61,35,C.white,true);
foot(s,'Synthetic station-opening experiment · not an observed causal effect in Baku',true);
note(s,`Fresh ATT ${e.did.att_log}; 95% CI [${e.did.ci_low}, ${e.did.ci_high}] log units. Do not confuse the planted fully ramped .08 with the approximate average post-opening .075 effect. Treated listings within 1 km, controls 2–6 km away. Cell and month fixed effects; clustered errors. Planted ramp lasts three months; see synthetic_truth.json. Source: research-kdtree/did_results.json; bakuml/causal/did.py. No real station causal conclusion is being made.`);
}
{
let s=slide('Future potential is a scenario, with assumptions','Polycentric scenario · projected 24-month appreciation (%) · synthetic H3 run');
await img(s,'growth-map.png',60,180,1160,480,'contain');
foot(s,'Ridge forecast fallback + estimated node adjustment · conditional research output');
note(s,'This map is generated from actual H3 scenario predictions, not a hand-painted map. The polycentric scenario doubles the estimated historical node-gravity appreciation slope. The pipeline here used SpatialLagRidgeForecaster because torch is not installed. These are conditional projections, not validated future market outcomes or a forecast of physical city expansion. Source: research-h3/predictions.parquet and metrics.json; bakuml/pipeline.py.');
}
{
let s=slide('This central cell projects a lower price','Three conditional scenarios · cell selected by location and sufficient history');
let names=['baseline','polycentric','transit'],labels=['Baseline','Polycentric','Transit'],last=e.example.history.at(-1).price_azn_m2_median;
chart(s,'bar',['12 months','24 months'],names.map((sc,i)=>({name:labels[i],values:[12,24].map(h=>round(e.example.predictions.find(r=>r.scenario===sc&&r.horizon_months===h).q50)),fill:[C.teal,C.coral,'#80A8BE'][i]})),{w:830,max:Math.ceil(Math.max(...e.example.predictions.map(r=>r.q50))/1000)*1000,unit:'Projected asking price (AZN/m²)'});
text(s,Math.round(last).toLocaleString('en-US'),938,275,270,96,58,C.teal,true);text(s,'AZN/m²\nlast observed median',940,380,263,100,24);
foot(s,'Synthetic cell example · forecast endpoints, not observed future prices');
note(s,`Cell ${e.example.cell}, centroid ${e.example.lat}, ${e.example.lon}, latest observed month ${e.example.history.at(-1).month}. Chosen as nearest eligible cell to the configured centre, not by forecast growth. Last observed ${last}. All six q50 endpoints plotted directly from predictions.parquet. Ridge fallback, no new claim of STGCN performance. Forecast intervals scale one-month relative conformal widths by square root of horizon; long-horizon coverage is not established. Source: evidence.json example; bakuml/pipeline.py.`);
}
{
let s=slide('The research is explorable in the software','Actual local application · scenario and horizon controls · cell-level map');
await img(s,'software.png',60,195,1160,456,'contain');
foot(s,'Captured with synthetic H3 artifacts and a visible research-demo label');
note(s,'Actual Flask application screenshot. A local capture wrapper serves research-h3 artifacts, adds a visible synthetic-demo banner, and converts the API’s RGBA palette arrays to CSS hex colours so the map renders. No production app source was changed. The screenshot illustrates current interaction, not a live deployed real-data product. Sources: app/flask_app.py; app/static/js/map.js; .presentation-build/demo_server.py.');
}
{
let cov=100*m.conformal.coverage_q10_q90;
let s=slide('Uncertainty still needs improvement','Observed interval coverage on three unseen months');
chart(s,'bar',['80% target','Observed'],[{name:'Coverage',values:[80,Number(cov.toFixed(1))],fill:C.teal}],{max:100,unit:'Coverage (%)'});
text(s,(80-cov).toFixed(1)+' pp',883,273,325,98,64,C.coral,true);text(s,'below the target',888,384,308,60,28);
text(s,'Long-horizon coverage\nis not yet established.',888,502,306,110,25);
foot(s,'Fresh synthetic KD-tree run · holdout: June–August 2026');
note(s,`Coverage ${cov}% versus80%, gap ${80-cov} percentage points. Six preceding calibration months and three held-out months. This updates the old README figure rather than repeating it. One-month interval coverage and long-horizon heuristic widths must not be conflated. Sources: metrics.json conformal; bakuml/models/conformal.py; bakuml/pipeline.py.`);
}
{
let s=slide('What our work establishes', '',C.dark);
text(s,'Geography matters.',65,216,1120,76,51,C.white,true);
text(s,'The monthly model beats persistence.',65,327,1120,76,43,C.pale,true);
text(s,'Scenarios make future assumptions visible.',65,438,1135,78,40,C.white,true);
text(s,'Next: longitudinal real data and 12/24-month backtests.',68,586,1115,62,25,C.pale);
note(s,'Conclusions are scoped to this synthetic research run: geography is a major fitted signal, XGBoost outperforms persistence in time and spatial tests, and the scenario engine produces explicit conditional futures. The real snapshot pipeline is separate and uses heuristic appreciation. To make credible real-market forecasting claims, collect repeated snapshots, compute neighbour features within folds, separate calibration and evaluation, and test each long horizon. Sources: pipeline.py; scripts/run_real_pipeline.py; metrics.json.');
}
{
let s=slide('BakuML','Holberton School · Group 3');
text(s,'We connect place, time and infrastructure\nto explore Baku’s future real estate potential.',64,230,1147,173,45,C.ink,true);
text(s,'Hüseyn Hüseynli · Nihad Süleymanov · Eldəniz Arifzadə\nZeynəb Mirzəzadə · Vüqar Dünyamalıyev',66,499,1145,100,25,C.muted);
text(s,'Questions?',66,635,600,46,27,C.teal,true);
note(s,'Team names reproduced from README.md. The central contribution is a spatial research pipeline, scenario outputs and an interface to explore them. Avoid describing this as a validated prediction of city land growth or investment returns.');
}
await (await PresentationFile.exportPptx(p)).save(B+'/candidate-v2.pptx');
for(let i=0;i<p.slides.items.length;i++){const b=await p.slides.items[i].export({format:'png',scale:1});await fs.writeFile(B+`/v2-${i+1}.png`,new Uint8Array(await b.arrayBuffer()));}
console.log('12-slide candidate and previews created');

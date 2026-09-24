import sys,json
from pathlib import Path
root=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(root/'.presentation-build/research-deps'),str(root)]
import pandas as pd,numpy as np,folium
from bakuml.viz import load_artifacts,hex_layer_geojson
from bakuml import config
out=root/'.presentation-build'
h3=load_artifacts(out/'research-h3'); kd=load_artifacts(out/'research-kdtree')
panel=kd['panel'];pred=kd['predictions']; feats=kd['cell_features'].set_index('h3')
# Select a central cell by geometry and adequate history, without selecting a desired forecast.
counts=panel.groupby('h3').size(); eligible=counts[counts>=30].index
central=feats.loc[feats.index.intersection(eligible)].sort_values('dist_centre_km').index[0]
hist=panel[panel.h3==central].sort_values('month')
fc=pred[pred.h3==central].sort_values(['scenario','horizon_months'])
evidence={'kd_metrics':kd['metrics'],'h3_metrics':h3['metrics'],'shap':kd['shap'],'did':kd['did'],
 'example':{'cell':central,'lat':float(feats.loc[central,'lat']),'lon':float(feats.loc[central,'lon']),'history':hist[['month','price_azn_m2_median']].to_dict('records'),'predictions':fc.to_dict('records')},
 'raw_dataset':{'path':'data/raw/listings.parquet','rows':len(pd.read_parquet(root/'data/raw/listings.parquet'))}}
(out/'evidence.json').write_text(json.dumps(evidence,indent=2),encoding='utf-8')
for filename,scen,metric in [('price-map','baseline','current_price'),('growth-map','polycentric','appreciation_pct')]:
 geo,cmap=hex_layer_geojson(h3['predictions'],h3['panel'],scenario=scen,horizon=24,metric=metric,geometry=h3['cell_geometry'])
 m=folium.Map(location=[40.455,49.91],zoom_start=11,tiles='OpenStreetMap',height=630,zoom_control=False,control_scale=True)
 folium.GeoJson(geo,style_function=lambda f:{'fillColor':cmap(f['properties']['value']),'color':'#344d5a','weight':0.45,'fillOpacity':0.76},tooltip=folium.GeoJsonTooltip(fields=['label'],aliases=[''])).add_to(m)
 cmap.caption=('Latest available synthetic median asking price (AZN/m²)' if metric=='current_price' else 'Synthetic polycentric scenario, 24-month projected appreciation (%)')
 cmap.add_to(m)
 m.save(str(out/(filename+'.html')))
print(json.dumps({'kd':kd['metrics']['dataset'],'wf':kd['metrics']['walk_forward']['aggregate'],'naive':kd['metrics']['walk_forward']['naive'],'spatial':kd['metrics']['spatial_cv']['aggregate'],'coverage':kd['metrics']['conformal'],'shap':kd['shap'],'cell':central,'predictions':fc.to_dict('records')},indent=2))

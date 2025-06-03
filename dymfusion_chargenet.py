import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader
import traceback # For detailed exception traceback
from math import radians, sin, cos, sqrt, atan2 # For Haversine distance
from torch.optim.lr_scheduler import ReduceLROnPlateau # Import ReduceLROnPlateau
import matplotlib.pyplot as plt
import seaborn as sns
import os # For creating directories

# --- PyTorch Geometric Check ---
try:
    from torch_geometric.nn import GATConv
    PYG_AVAILABLE = True
    print("PyTorch Geometric found and loaded.")
except ImportError:
    PYG_AVAILABLE = False
    print("Warning: PyTorch Geometric not found. GATConv will be a placeholder.")
    class GATConv(nn.Module): # Placeholder
        def __init__(self, in_channels, out_channels, heads=1, concat=True, dropout=0.0, **kwargs):
            super().__init__();
            self.is_concat_output = concat if heads > 0 else False
            effective_out_channels = out_channels * heads if self.is_concat_output else out_channels
            self.lin = nn.Linear(in_channels, effective_out_channels)
            self.heads = heads
            self.out_channels_per_head = out_channels
            if not PYG_AVAILABLE:
                print(f"Using Placeholder GATConv: in={in_channels}, out_per_head={out_channels}, heads={heads}, concat={self.is_concat_output}, effective_out={effective_out_channels}")

        def forward(self, x, edge_index, size=None, return_attention_weights=None):
            if x.dim() == 0 or (x.numel() > 0 and x.shape[-1] == 0):
                 if self.is_concat_output: return torch.zeros(x.shape[0], self.out_channels_per_head * self.heads, device=x.device)
                 else: return torch.zeros(x.shape[0], self.out_channels_per_head, device=x.device)
            if x.numel() > 0 and x.shape[-1] != self.lin.in_features:
                raise ValueError(f"Placeholder GATConv: Expected input features {self.lin.in_features}, got {x.shape[-1]}")
            x_transformed = self.lin(x) if x.numel() > 0 else torch.zeros(x.shape[0], self.lin.out_features, device=x.device)
            if return_attention_weights:
                num_edges = edge_index.shape[1] if edge_index is not None and edge_index.numel() > 0 else 0
                att_weights = (edge_index, torch.zeros(num_edges, self.heads, device=x.device))
                return x_transformed, att_weights
            return x_transformed

# --- Configuration ---
DATA_PATH = 'UrbanEV-main/data/'
TARGET_COLUMN = 'occupancy'
HIST_SEQ_LEN = 24
PRED_SEQ_LEN = 3
TARGET_DIM = 1
MAFE_EMBED_DIM = 32
STGA_GAT_OUT_CHANNELS_PER_HEAD = 16
STGA_GAT_HEADS = 2
STGA_LSTM_HIDDEN_DIM = 64
STGA_OUT_DIM = 64
CDYN_DIM = 16
FUSED_DIM = 128
PD_HIDDEN_DIM = 64
BATCH_SIZE = 32
EPOCHS = 50
LR = 0.001
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
DISTANCE_THRESHOLD_KM = 2.0
PATIENCE_EARLY_STOPPING = 10
SCHEDULER_PATIENCE = 5

DEFAULT_ABLATION_CONFIG = {
    "name": "default_full_model", "use_poi": True, "use_weather_mafe": True,
    "event_rt_column_names_to_use": None, "disable_stga": False, "disable_dcf": False
}

# --- Haversine Distance & Shock Simulation ---
def haversine_distance(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2]); dlon = lon2-lon1; dlat = lat2-lat1
    a = sin(dlat/2)**2 + cos(lat1)*cos(lat2)*sin(dlon/2)**2; c = 2*atan2(sqrt(a),sqrt(1-a)); r=6371
    return r*c

def simulate_dynamic_shocks(df_target_scaled_input, df_event_rt_input, shock_config=None):
    if shock_config is None: return df_target_scaled_input, df_event_rt_input
    df_target_shocked = df_target_scaled_input.copy(); df_event_rt_updated = df_event_rt_input.copy()
    num_timesteps, num_zones = df_target_shocked.shape
    for shock_type, params in shock_config.items():
        print(f"Simulating shock: {shock_type}..."); shock_indicator_col = f"shock_{shock_type}"
        df_event_rt_updated[shock_indicator_col] = 0
        num_zones_to_affect = max(1, int(num_zones * params.get("num_zones_affected_ratio", 0.1)))
        t = 0
        while t < (num_timesteps - params["duration_hours"]):
            if np.random.rand() < params["prob"]:
                start_time_idx=t; end_time_idx=min(t+params["duration_hours"], num_timesteps)
                affected_zone_indices = np.random.choice(num_zones, num_zones_to_affect, replace=False)
                impact_direction=1 if shock_type=="large_event" else -1
                shock_magnitude=np.random.uniform(0.5,1.0)*params["impact_factor"]*impact_direction
                df_target_shocked.iloc[start_time_idx:end_time_idx, affected_zone_indices] += shock_magnitude
                df_event_rt_updated.loc[df_event_rt_updated.index[start_time_idx:end_time_idx], shock_indicator_col] = 1.0
                t += params["duration_hours"]
            else: t += 1
    return df_target_shocked, df_event_rt_updated

# --- UrbanEV Data Loading and Preprocessing ---
def load_and_preprocess_urban_ev(data_path, hist_seq_len, pred_seq_len, target_column='occupancy', ablation_cfg=None):
    current_ablation_cfg = ablation_cfg if ablation_cfg is not None else DEFAULT_ABLATION_CONFIG.copy()
    print(f"Data loading with ablation config: {current_ablation_cfg.get('name', 'N/A')}")
    try:
        df_target=pd.read_csv(f"{data_path}{target_column}.csv",index_col=0); df_adj_raw=pd.read_csv(f"{data_path}adj.csv",index_col=0)
        df_weather_airport=pd.read_csv(f"{data_path}weather_airport.csv",parse_dates=['time'],index_col='time')
        df_inf=pd.read_csv(f"{data_path}inf.csv",index_col=0); df_poi_raw=pd.read_csv(f"{data_path}poi.csv")
    except Exception as e: print(f"Error loading file: {e}"); return None,None,None,None,None,{},None
    df_target.index=pd.to_datetime(df_target.index)
    weather_features_to_use=['T','P0','P','U','nRAIN','Td']
    actual_weather_cols=[col for col in weather_features_to_use if col in df_weather_airport.columns]
    if not actual_weather_cols: print("CRIT ERR: No valid weather cols."); return None,None,None,None,None,{},None
    df_weather=df_weather_airport[actual_weather_cols].copy()
    for col in df_weather.columns:
        if not pd.api.types.is_numeric_dtype(df_weather[col]):
            try: df_weather[col]=pd.to_numeric(df_weather[col],errors='coerce'); df_weather[col]=df_weather[col].fillna(df_weather[col].median())
            except ValueError: print(f"CRIT ERR: Weather col '{col}' not numeric."); return None,None,None,None,None,{},None
    df_weather=df_weather.fillna(df_weather.median())
    common_index=df_target.index.intersection(df_weather.index)
    if common_index.empty: print("CRIT ERR: No common time index."); return None,None,None,None,None,{},None
    df_target=df_target.loc[common_index].ffill().bfill(); df_weather=df_weather.loc[common_index].ffill().bfill()
    num_zones=df_target.shape[1]
    if num_zones==0 or len(df_target)==0: print("ERR: No zones/data after align."); return None,None,None,None,None,{},None
    target_scaler=StandardScaler(); df_target_scaled=pd.DataFrame(target_scaler.fit_transform(df_target),index=df_target.index,columns=df_target.columns)
    weather_scaler=StandardScaler(); df_weather_scaled=pd.DataFrame(weather_scaler.fit_transform(df_weather),index=df_weather.index,columns=df_weather.columns)
    STATIC_ATTR_DIM_ACTUAL=2; static_attr_list=[]
    df_inf_index_str=df_inf.index.astype(str)
    for zone_id_str_target in df_target.columns.astype(str):
        if zone_id_str_target in df_inf_index_str:
            inf_series=df_inf.loc[df_inf_index_str==zone_id_str_target].iloc[0]
            cap=inf_series.get('charging_capacity_total',0); area=inf_series.get('area',0)
            static_attr_list.append([cap,area])
        else: static_attr_list.append([0,0])
    static_attr_np_raw=np.array(static_attr_list,dtype=np.float32)
    if static_attr_np_raw.shape[0]==num_zones: static_attr_np=StandardScaler().fit_transform(static_attr_np_raw)
    else: static_attr_np=np.zeros((num_zones,STATIC_ATTR_DIM_ACTUAL),dtype=np.float32)
    print(f"Static attributes processed. Shape: {static_attr_np.shape}")
    poi_features_np=np.zeros((num_zones,3),dtype=np.float32); POI_DIM_ACTUAL=3
    zone_coordinates = {}
    if current_ablation_cfg.get("use_poi",True):
        print("Processing POI features..."); POI_RADIUS_KM=0.5; EXPECTED_POI_CATEGORIES=sorted(['food and beverage services','business and residential','lifestyle services'])
        POI_DIM_ACTUAL=len(EXPECTED_POI_CATEGORIES); poi_features_list=[]
        df_inf_index_str_for_poi=df_inf.index.astype(str)
        for zone_id_str_inf in df_inf_index_str_for_poi:
            inf_series=df_inf.loc[df_inf_index_str_for_poi==zone_id_str_inf].iloc[0]
            if 'longitude' in inf_series and 'latitude' in inf_series and pd.notna(inf_series['longitude']) and pd.notna(inf_series['latitude']):
                zone_coordinates[zone_id_str_inf]=(float(inf_series['longitude']),float(inf_series['latitude']))
        if not ('longitude' in df_poi_raw.columns and 'latitude' in df_poi_raw.columns and 'primary_types' in df_poi_raw.columns):
            print("CRIT ERR: poi.csv missing cols. Zeros for POI."); poi_features_np=np.zeros((num_zones,POI_DIM_ACTUAL),dtype=np.float32)
        else:
            for zone_id_str_target in df_target.columns.astype(str):
                zone_poi_counts={cat:0 for cat in EXPECTED_POI_CATEGORIES}
                if zone_id_str_target in zone_coordinates:
                    zone_lon,zone_lat=zone_coordinates[zone_id_str_target]
                    for _,poi_row in df_poi_raw.iterrows():
                        if pd.notna(poi_row['longitude']) and pd.notna(poi_row['latitude']):
                            distance=haversine_distance(zone_lon,zone_lat,poi_row['longitude'],poi_row['latitude'])
                            if distance<=POI_RADIUS_KM:
                                poi_type_str=str(poi_row['primary_types']).strip()
                                if poi_type_str in zone_poi_counts: zone_poi_counts[poi_type_str]+=1
                poi_features_list.append([zone_poi_counts[cat] for cat in EXPECTED_POI_CATEGORIES])
            if poi_features_list:
                poi_features_np_raw=np.array(poi_features_list,dtype=np.float32)
                if poi_features_np_raw.shape[0]==num_zones and poi_features_np_raw.shape[1]==POI_DIM_ACTUAL:
                    poi_features_np=StandardScaler().fit_transform(poi_features_np_raw)
                else: poi_features_np=np.zeros((num_zones,POI_DIM_ACTUAL),dtype=np.float32); print("CRIT Warn: POI shape mismatch. Zeros.")
            else: poi_features_np=np.zeros((num_zones,POI_DIM_ACTUAL),dtype=np.float32); print("CRIT Warn: POI list empty. Zeros.")
        print(f"POI features processed. Shape: {poi_features_np.shape}")
    else: print("POI features SKIPPED."); POI_DIM_ACTUAL=0; zone_coordinates = {}
    df_event_rt_base=pd.DataFrame(index=df_weather_scaled.index)
    if 'T' in df_weather_scaled.columns:
        temp_s=df_weather_scaled['T']; high_t,low_t=temp_s.quantile(0.9),temp_s.quantile(0.1)
        df_event_rt_base['extreme_high_temp']=(temp_s>high_t).astype(int); df_event_rt_base['extreme_low_temp']=(temp_s<low_t).astype(int)
    else: df_event_rt_base['extreme_high_temp']=0; df_event_rt_base['extreme_low_temp']=0
    if 'nRAIN' in df_weather_scaled.columns:
        rain_s=df_weather_scaled['nRAIN']; rain_thresh=rain_s.quantile(0.75)
        df_event_rt_base['heavy_rain']=(rain_s>rain_thresh).astype(int)
    else: df_event_rt_base['heavy_rain']=0
    df_event_rt_base['is_weekend']=(df_event_rt_base.index.dayofweek>=5).astype(int)
    df_event_rt_base['is_morning_rush']=((df_event_rt_base.index.hour>=7)&(df_event_rt_base.index.hour<=9)).astype(int)
    df_event_rt_base['is_evening_rush']=((df_event_rt_base.index.hour>=17)&(df_event_rt_base.index.hour<=19)).astype(int)
    df_event_rt_base['is_night']=((df_event_rt_base.index.hour>=22)|(df_event_rt_base.index.hour<=5)).astype(int)
    df_event_rt_base['is_summer']=((df_event_rt_base.index.month>=6)&(df_event_rt_base.index.month<=8)).astype(int)
    df_event_rt_base['is_winter']=((df_event_rt_base.index.month==12)|(df_event_rt_base.index.month<=2)).astype(int)
    shock_sim_cfg={'traffic_jam':{"prob":0.02,"duration_hours":4,"impact_factor":0.4,"num_zones_affected_ratio":0.15},
                     'large_event':{"prob":0.015,"duration_hours":8,"impact_factor":0.6,"num_zones_affected_ratio":0.08}}
    df_target_final_for_seq,df_event_rt_with_shocks=simulate_dynamic_shocks(df_target_scaled,df_event_rt_base,shock_config=shock_sim_cfg)
    df_event_rt_final_for_model=df_event_rt_with_shocks
    event_rt_cols_to_use=current_ablation_cfg.get("event_rt_column_names_to_use")
    if event_rt_cols_to_use is not None:
        valid_subset_cols=[col for col in event_rt_cols_to_use if col in df_event_rt_with_shocks.columns]
        if len(valid_subset_cols)<len(event_rt_cols_to_use): print(f"Warn: Not all event_rt_cols found. Using: {valid_subset_cols}")
        if not valid_subset_cols: print("Warn: event_rt_cols empty. No event_rt feats."); df_event_rt_final_for_model=pd.DataFrame(index=df_event_rt_with_shocks.index)
        else: df_event_rt_final_for_model=df_event_rt_with_shocks[valid_subset_cols]
    EVENT_RT_DIM_ACTUAL=df_event_rt_final_for_model.shape[1]
    print(f"Final event_rt feats for model (scen: {current_ablation_cfg.get('name')}): {list(df_event_rt_final_for_model.columns)}")
    if EVENT_RT_DIM_ACTUAL==0 and not current_ablation_cfg.get("disable_dcf",False): print("Warn: DCF active but no event_rt feats selected.")
    X_hd_l,X_w_l,X_p_l,X_sa_l,X_er_l,Y_l=[],[],[],[],[],[]
    min_len_seq=hist_seq_len+pred_seq_len
    if len(df_target_final_for_seq)<min_len_seq: print(f"CRIT: Data len too short for seq."); return None,None,None,None,None,{},None
    for i in range(len(df_target_final_for_seq)-min_len_seq+1):
        X_hd_l.append(df_target_scaled.iloc[i:i+hist_seq_len].values.reshape(hist_seq_len,num_zones,TARGET_DIM))
        X_w_l.append(df_weather_scaled.iloc[i:i+hist_seq_len].values)
        X_p_l.append(poi_features_np); X_sa_l.append(static_attr_np)
        if EVENT_RT_DIM_ACTUAL>0: X_er_l.append(df_event_rt_final_for_model.iloc[i+hist_seq_len-1].values)
        else: X_er_l.append(np.array([],dtype=np.float32))
        Y_l.append(df_target_final_for_seq.iloc[i+hist_seq_len:i+hist_seq_len+pred_seq_len].values.reshape(pred_seq_len,num_zones,TARGET_DIM))
    if not Y_l: print("ERR: No sequences."); return None,None,None,None,None,{},None
    data_X={"hist_demand":np.array(X_hd_l,dtype=np.float32),"weather":np.array(X_w_l,dtype=np.float32),
            "poi":np.array(X_p_l,dtype=np.float32),"static_attr":np.array(X_sa_l,dtype=np.float32),
            "event_rt_current":np.array(X_er_l,dtype=np.float32)}
    if EVENT_RT_DIM_ACTUAL==0: data_X["event_rt_current"]=np.empty((len(X_er_l),0),dtype=np.float32)
    data_Y=np.array(Y_l,dtype=np.float32)
    print("Processing Adjacency Matrix...")
    target_zone_ids_str_list=df_target.columns.astype(str).tolist(); adj_raw_index_str_list=df_adj_raw.index.astype(str).tolist()
    adj_raw_cols_str_list=df_adj_raw.columns.astype(str).tolist(); target_id_to_idx_map={zone_id:i for i,zone_id in enumerate(target_zone_ids_str_list)}
    adj_matrix_from_file_np=np.zeros((num_zones,num_zones),dtype=np.float32); num_connections_from_file=0
    for r_idx_adj,zone_id_row_adj in enumerate(adj_raw_index_str_list):
        if zone_id_row_adj in target_id_to_idx_map:
            target_row_idx=target_id_to_idx_map[zone_id_row_adj]
            for c_idx_adj,zone_id_col_adj in enumerate(adj_raw_cols_str_list):
                if zone_id_col_adj in target_id_to_idx_map:
                    target_col_idx=target_id_to_idx_map[zone_id_col_adj]
                    adj_value=df_adj_raw.iloc[r_idx_adj,c_idx_adj]
                    if pd.notna(adj_value) and adj_value>0:
                        adj_matrix_from_file_np[target_row_idx,target_col_idx]=float(adj_value)
                        if target_row_idx!=target_col_idx: num_connections_from_file+=1
    np.fill_diagonal(adj_matrix_from_file_np,1)
    print(f"Original aligned adj created. Shape: {adj_matrix_from_file_np.shape}. Found {num_connections_from_file} inter-zone connections.")
    adj_matrix_to_use_np=adj_matrix_from_file_np
    if num_connections_from_file==0 and num_zones>1 and not current_ablation_cfg.get("disable_stga",False):
        print("Original adj has no inter-zone connections. Attempting distance-based...")
        if len(zone_coordinates)>=0.8*num_zones:
            distance_adj_matrix=np.zeros((num_zones,num_zones),dtype=np.float32); distance_connections=0
            for i,zone_id_i in enumerate(target_zone_ids_str_list):
                if zone_id_i in zone_coordinates:
                    lon_i,lat_i=zone_coordinates[zone_id_i]
                    for j,zone_id_j in enumerate(target_zone_ids_str_list):
                        if i!=j and zone_id_j in zone_coordinates:
                            lon_j,lat_j=zone_coordinates[zone_id_j]
                            try:
                                dist=haversine_distance(lon_i,lat_i,lon_j,lat_j)
                                if dist<DISTANCE_THRESHOLD_KM: distance_adj_matrix[i,j]=1.0/(dist+0.1); distance_connections+=1
                            except Exception as e_dist: print(f"Err calc dist {zone_id_i}-{zone_id_j}: {e_dist}")
            np.fill_diagonal(distance_adj_matrix,1.0)
            print(f"Distance-based adj created with {distance_connections} connections (thresh: {DISTANCE_THRESHOLD_KM}km).")
            if distance_connections>0: adj_matrix_to_use_np=distance_adj_matrix; print("Using distance-based adj.")
            else: print("Warn: Dist-based adj also no connections. Using original.")
        else: print(f"Warn: Not enough coords ({len(zone_coordinates)}/{num_zones}) for dist-adj. Using original.")
    elif current_ablation_cfg.get("disable_stga",False): print("STGA disabled, using identity matrix for 'adj'."); adj_matrix_to_use_np = np.eye(num_zones, dtype=np.float32)
    edge_index=torch.empty((2,0),dtype=torch.long)
    if PYG_AVAILABLE and not current_ablation_cfg.get("disable_stga",False):
        try:
            adj_tensor_for_edges=torch.from_numpy(adj_matrix_to_use_np)
            edge_indices_nonzero=(adj_tensor_for_edges>0).nonzero(as_tuple=False)
            if edge_indices_nonzero.numel()>0:
                edge_index_candidate=edge_indices_nonzero.t().contiguous()
                if edge_index_candidate.shape[0]==2: edge_index=edge_index_candidate; print(f"Created edge_index shape {edge_index.shape}")
                else: print(f"Warn: Gen edge_index incorrect shape: {edge_index_candidate.shape}. Empty.")
            else: print("Warn: No edges in final adj. edge_index empty.")
        except Exception as e_edge: print(f"Err creating edge_index: {e_edge}. Empty.")
    graph_features={"adj":adj_matrix_to_use_np,"edge_index":edge_index}
    actual_dims={"hist_demand_dim":data_X["hist_demand"].shape[-1],"weather_dim":df_weather_scaled.shape[1],
                 "poi_dim":POI_DIM_ACTUAL,"static_attr_dim":STATIC_ATTR_DIM_ACTUAL,
                 "event_rt_dim":EVENT_RT_DIM_ACTUAL,"num_zones":num_zones,
                 "event_rt_column_names":list(df_event_rt_final_for_model.columns)}
    print(f"Actual feature dimensions for model: {actual_dims}")
    n_total=data_Y.shape[0];
    if n_total<3: print("CRIT ERR: Not enough samples for split."); return None,None,None,None,None,{},None
    train_len=int(n_total*TRAIN_RATIO); val_len=int(n_total*VAL_RATIO)
    if train_len==0 and n_total>0: train_len=1
    if val_len==0 and (n_total-train_len)>0: val_len=1
    test_len=n_total-train_len-val_len
    if test_len<0: val_len+=test_len; test_len=0
    if val_len<0: val_len=0
    if train_len+val_len+test_len>n_total:
        val_len=n_total-train_len-test_len
        if val_len<0: val_len=0; test_len=n_total-train_len
    if val_len==0 and test_len>0 and (n_total-train_len-test_len>=1):
        val_len=1; test_len-=1
        if test_len<0: test_len=0
    train_end_idx=train_len; val_end_idx=train_len+val_len
    train_X={k:v[:train_end_idx] for k,v in data_X.items()}; train_Y=data_Y[:train_end_idx]
    val_X={k:v[train_end_idx:val_end_idx] for k,v in data_X.items()}; val_Y=data_Y[train_end_idx:val_end_idx]
    test_X={k:v[val_end_idx:] for k,v in data_X.items()}; test_Y=data_Y[val_end_idx:]
    print(f"Data split: Total {n_total}, Train {len(train_Y)}, Val {len(val_Y)}, Test {len(test_Y)}")
    if len(val_Y)==0 and n_total>len(train_Y) and n_total>1: print("CRIT WARN: val_Y empty.")
    df_event_rt_test_for_analysis=None
    if test_X["event_rt_current"].shape[0]>0 and actual_dims["event_rt_dim"]>0:
        df_event_rt_test_for_analysis=pd.DataFrame(test_X["event_rt_current"],columns=actual_dims["event_rt_column_names"])
    elif test_X["event_rt_current"].shape[0]>0 and actual_dims["event_rt_dim"]==0:
        df_event_rt_test_for_analysis=pd.DataFrame(index=range(test_X["event_rt_current"].shape[0]))
    return (train_X,train_Y),(val_X,val_Y),(test_X,test_Y),graph_features,target_scaler,actual_dims,df_event_rt_test_for_analysis

# --- Model Definitions ---
class MAFESubEncoder(nn.Module):
    def __init__(self, input_dim, output_dim, is_sequence_input=False, has_station_dim=False):
        super().__init__()
        self.fc = nn.Linear(input_dim, output_dim) if input_dim > 0 else None
        self.is_sequence_input, self.has_station_dim = is_sequence_input, has_station_dim
        self.input_dim = input_dim; self.output_dim = output_dim
    def forward(self, x):
        if self.fc is None:
            B=x.shape[0]; S=x.shape[1] if self.has_station_dim and x.dim()>(2 if not self.is_sequence_input else 3) else -1
            if self.has_station_dim: return torch.zeros(B,S if S!=-1 else x.shape[1],self.output_dim,device=x.device)
            else: return torch.zeros(B,self.output_dim,device=x.device)
        if self.has_station_dim and self.is_sequence_input:B,S,T,F_in=x.shape;x_processed=self.fc(x.reshape(B*S,T,F_in));x_output=x_processed[:,-1,:].view(B,S,-1)
        elif not self.has_station_dim and self.is_sequence_input:x_processed=self.fc(x);x_output=x_processed[:,-1,:]
        elif self.has_station_dim and not self.is_sequence_input:B,S,F_in=x.shape;x_processed=self.fc(x.reshape(B*S,F_in));x_output=x_processed.view(B,S,-1)
        elif not self.has_station_dim and not self.is_sequence_input:x_output=self.fc(x)
        else: raise ValueError(f"Unhandled MAFE input: {x.shape}, seq={self.is_sequence_input}, station={self.has_station_dim}")
        return torch.relu(x_output)

class STGA(nn.Module):
    def __init__(self, node_feat_dim_for_gat, gat_out_channels_per_head, gat_heads, lstm_hidden_dim, stga_out_dim, num_stations):
        super().__init__(); self.num_stations = num_stations; self.gat_concat_flag = True if gat_heads > 0 else False
        if PYG_AVAILABLE:
            actual_gat_heads=gat_heads if gat_heads>0 else 1
            self.gat1=GATConv(node_feat_dim_for_gat,gat_out_channels_per_head,heads=actual_gat_heads,concat=self.gat_concat_flag)
            self.gat_out_dim_effective=gat_out_channels_per_head*actual_gat_heads if self.gat_concat_flag else gat_out_channels_per_head
        else:
            self.gat1=GATConv(node_feat_dim_for_gat,gat_out_channels_per_head*(gat_heads if gat_heads>0 else 1),heads=1,concat=False)
            self.gat_out_dim_effective=gat_out_channels_per_head*(gat_heads if gat_heads>0 else 1)
        self.lstm=nn.LSTM(self.gat_out_dim_effective,lstm_hidden_dim,batch_first=True); self.fc_out=nn.Linear(lstm_hidden_dim,stga_out_dim)
    def forward(self, hist_demand_seq, static_attr, edge_index_shared):
        B,T_hist,S_nodes,D_hd_raw=hist_demand_seq.shape; gat_outputs_over_time=[]
        valid_empty_edge_index=torch.empty((2,0),dtype=torch.long,device=hist_demand_seq.device)
        current_edge_index_for_gat=edge_index_shared
        if not (current_edge_index_for_gat is not None and isinstance(current_edge_index_for_gat,torch.Tensor) and \
                current_edge_index_for_gat.numel()>0 and current_edge_index_for_gat.dim()==2 and \
                current_edge_index_for_gat.shape[0]==2): current_edge_index_for_gat=valid_empty_edge_index
        for t_step in range(T_hist):
            curr_hist_demand_t=hist_demand_seq[:,t_step,:,:]; combined_node_features_for_gat_t=torch.cat([curr_hist_demand_t,static_attr],dim=-1)
            batch_gat_outputs_t=[]
            for b_idx in range(B):
                nodes_for_gat_b_t=combined_node_features_for_gat_t[b_idx]
                try: gat_out_b_t=self.gat1(nodes_for_gat_b_t,current_edge_index_for_gat)
                except Exception: gat_out_b_t=torch.zeros(S_nodes,self.gat_out_dim_effective,device=nodes_for_gat_b_t.device)
                batch_gat_outputs_t.append(gat_out_b_t)
            gat_outputs_over_time.append(torch.stack(batch_gat_outputs_t))
        gat_sequence_tensor=torch.stack(gat_outputs_over_time,dim=1)
        lstm_input=gat_sequence_tensor.permute(0,2,1,3).reshape(B*self.num_stations,T_hist,self.gat_out_dim_effective)
        _,(hn,_)=self.lstm(lstm_input); last_layer_hidden_state=hn[-1].view(B,self.num_stations,-1)
        return torch.relu(self.fc_out(last_layer_hidden_state))

class DCF(nn.Module):
    def __init__(self, mafe_embed_dims_dict, stga_out_dim, cdyn_dim, fused_dim):
        super().__init__(); self.mafe_gates=nn.ModuleDict(); self.active_mafe_keys=[k for k,d in mafe_embed_dims_dict.items() if d>0]
        fusion_input_dim_calculated=0
        for key in self.active_mafe_keys:
            dim=mafe_embed_dims_dict[key]; self.mafe_gates[key]=nn.Linear(dim+cdyn_dim,dim); fusion_input_dim_calculated+=dim
        self.stga_gate=None
        if stga_out_dim>0 and (stga_out_dim+cdyn_dim>0): self.stga_gate=nn.Linear(stga_out_dim+cdyn_dim,stga_out_dim); fusion_input_dim_calculated+=stga_out_dim
        if fusion_input_dim_calculated==0: self.fusion_mlp=nn.Linear(1,fused_dim); self.is_dummy_fusion=True
        else: self.fusion_mlp=nn.Sequential(nn.Linear(fusion_input_dim_calculated,fused_dim*2),nn.ReLU(),nn.Linear(fused_dim*2,fused_dim)); self.is_dummy_fusion=False
    def forward(self,mafe_outputs_dict,stga_output,cdyn_vector,num_stations):
        B,gated_features_for_concat=cdyn_vector.shape[0],[]
        cdyn_expanded_station=cdyn_vector.unsqueeze(1).repeat(1,num_stations,1)
        for key in self.active_mafe_keys:
            if key not in mafe_outputs_dict or mafe_outputs_dict[key] is None: continue
            mafe_feat_single_source=mafe_outputs_dict[key]
            if mafe_feat_single_source.dim()==2: mafe_feat_expanded_station=mafe_feat_single_source.unsqueeze(1).repeat(1,num_stations,1)
            elif mafe_feat_single_source.dim()==3 and mafe_feat_single_source.shape[1]==num_stations: mafe_feat_expanded_station=mafe_feat_single_source
            else: raise ValueError(f"Unexpected MAFE shape '{key}': {mafe_feat_single_source.shape}")
            gate_input_mafe=torch.cat([mafe_feat_expanded_station,cdyn_expanded_station],dim=-1)
            gated_mafe_feature=torch.sigmoid(self.mafe_gates[key](gate_input_mafe))*mafe_feat_expanded_station
            gated_features_for_concat.append(gated_mafe_feature)
        if self.stga_gate and stga_output is not None and stga_output.numel()>0:
            gate_input_stga=torch.cat([stga_output,cdyn_expanded_station],dim=-1)
            gated_stga_feature=torch.sigmoid(self.stga_gate(gate_input_stga))*stga_output
            gated_features_for_concat.append(gated_stga_feature)
        if not gated_features_for_concat:
            if self.is_dummy_fusion: return torch.relu(self.fusion_mlp(torch.zeros(B,num_stations,1,device=cdyn_vector.device)))
            else: return torch.zeros(B,num_stations,self.fusion_mlp[-1].out_features,device=cdyn_vector.device)
        return torch.relu(self.fusion_mlp(torch.cat(gated_features_for_concat,dim=-1)))

class PredictiveDecoder(nn.Module):
    def __init__(self,fused_dim,pd_hidden_dim,pred_seq_len,target_dim):
        super().__init__(); self.pred_seq_len,self.target_dim=pred_seq_len,target_dim
        self.mlp=nn.Sequential(nn.Linear(fused_dim,pd_hidden_dim),nn.ReLU(),nn.Linear(pd_hidden_dim,pred_seq_len*target_dim))
    def forward(self,fused_input_per_station):B,S,_=fused_input_per_station.shape;p=self.mlp(fused_input_per_station);return p.view(B,S,self.pred_seq_len,self.target_dim).permute(0,2,1,3)

class DyMFusionChargeNet(nn.Module):
    def __init__(self,config_model_hyperparams,actual_feature_dims,ablation_cfg=None):
        super().__init__(); self.config_hyperparams=config_model_hyperparams; self.num_stations=actual_feature_dims['num_zones']
        self.ablation_cfg=ablation_cfg if ablation_cfg is not None else DEFAULT_ABLATION_CONFIG.copy()
        print(f"Init DyMFusion with ablation: {self.ablation_cfg.get('name','N/A')}")
        self.mafe_hist_demand=MAFESubEncoder(actual_feature_dims['hist_demand_dim'],config_model_hyperparams['mafe_embed_dim'],True,True)
        weather_in_dim=actual_feature_dims['weather_dim'] if self.ablation_cfg.get("use_weather_mafe",True) else 0
        self.mafe_weather=MAFESubEncoder(weather_in_dim,config_model_hyperparams['mafe_embed_dim'],True,False) if weather_in_dim>0 else None
        poi_in_dim=actual_feature_dims['poi_dim'] if self.ablation_cfg.get("use_poi",True) else 0
        self.mafe_poi=MAFESubEncoder(poi_in_dim,config_model_hyperparams['mafe_embed_dim'],False,True) if poi_in_dim>0 else None
        encoder_dyn_in_dim=actual_feature_dims['event_rt_dim']
        self.encoder_dyn_context=MAFESubEncoder(encoder_dyn_in_dim,config_model_hyperparams['cdyn_dim'],False,False) if encoder_dyn_in_dim>0 else None
        if not self.ablation_cfg.get("disable_stga",False):
            stga_node_feat_dim=actual_feature_dims['hist_demand_dim']+actual_feature_dims['static_attr_dim']
            self.stga=STGA(stga_node_feat_dim,config_model_hyperparams['stga_gat_out_channels_per_head'],config_model_hyperparams['stga_gat_heads'],config_model_hyperparams['stga_lstm_hidden_dim'],config_model_hyperparams['stga_out_dim'],self.num_stations)
        else: self.stga=None; print(f"STGA DISABLED for {self.ablation_cfg.get('name')}")
        mafe_dims_for_fusion={'hist_demand':config_model_hyperparams['mafe_embed_dim']}
        if self.mafe_weather: mafe_dims_for_fusion['weather']=config_model_hyperparams['mafe_embed_dim']
        if self.mafe_poi: mafe_dims_for_fusion['poi']=config_model_hyperparams['mafe_embed_dim']
        stga_dim_for_fusion=config_model_hyperparams['stga_out_dim'] if self.stga else 0
        if not self.ablation_cfg.get("disable_dcf",False):
            self.dcf=DCF(mafe_dims_for_fusion,stga_dim_for_fusion,config_model_hyperparams['cdyn_dim'],config_model_hyperparams['fused_dim'])
            self.static_fusion_mlp=None; print(f"DCF ENABLED for {self.ablation_cfg.get('name')}")
        else:
            self.dcf=None; static_fusion_in_dim=sum(mafe_dims_for_fusion.values())+stga_dim_for_fusion
            if static_fusion_in_dim==0: self.static_fusion_mlp=nn.Linear(1,config_model_hyperparams['fused_dim']); print(f"WARN: Static MLP input 0 for {self.ablation_cfg.get('name')}. Dummy.")
            else: self.static_fusion_mlp=nn.Sequential(nn.Linear(static_fusion_in_dim,config_model_hyperparams['fused_dim']*2),nn.ReLU(),nn.Linear(config_model_hyperparams['fused_dim']*2,config_model_hyperparams['fused_dim']))
            print(f"DCF DISABLED (StaticFusion) for {self.ablation_cfg.get('name')}. Static MLP input: {static_fusion_in_dim}")
        self.pd=PredictiveDecoder(config_model_hyperparams['fused_dim'],config_model_hyperparams['pd_hidden_dim'],config_model_hyperparams['pred_seq_len'],config_model_hyperparams['target_dim'])
    def forward(self,inputs):
        B=inputs['hist_demand'].shape[0]; mafe_out_for_dcf={}; dev=inputs['hist_demand'].device
        f_hd=self.mafe_hist_demand(inputs['hist_demand'].permute(0,2,1,3)); mafe_out_for_dcf['hist_demand']=f_hd
        if self.mafe_weather: f_w=self.mafe_weather(inputs['weather']); mafe_out_for_dcf['weather']=f_w
        if self.mafe_poi: f_p=self.mafe_poi(inputs['poi']); mafe_out_for_dcf['poi']=f_p
        f_stga=torch.zeros(B,self.num_stations,self.config_hyperparams['stga_out_dim'],device=dev)
        if self.stga: f_stga=self.stga(inputs['hist_demand'],inputs['static_attr'],inputs['edge_index'])
        if self.dcf:
            cdyn_t=torch.zeros(B,self.config_hyperparams['cdyn_dim'],device=dev)
            if self.encoder_dyn_context and inputs['event_rt_current'].shape[-1]>0: cdyn_t=self.encoder_dyn_context(inputs['event_rt_current'])
            fused_rep=self.dcf(mafe_out_for_dcf,f_stga,cdyn_t,self.num_stations)
        else:
            s_fusion_inputs=[mafe_out_for_dcf['hist_demand']]
            if 'weather' in mafe_out_for_dcf and self.mafe_weather: s_fusion_inputs.append(mafe_out_for_dcf['weather'].unsqueeze(1).repeat(1,self.num_stations,1))
            if 'poi' in mafe_out_for_dcf and self.mafe_poi: s_fusion_inputs.append(mafe_out_for_dcf['poi'])
            s_fusion_inputs.append(f_stga)
            if not s_fusion_inputs: fused_rep=torch.relu(self.static_fusion_mlp(torch.zeros(B,self.num_stations,1,device=dev)))
            else: fused_rep=torch.relu(self.static_fusion_mlp(torch.cat(s_fusion_inputs,dim=-1)))
        return self.pd(fused_rep)

class LSTMBaseline(nn.Module):
    def __init__(self,input_dim_per_station_actual,lstm_hidden_dim,pred_seq_len,target_dim,num_stations_actual):
        super().__init__(); self.lstm=nn.LSTM(input_dim_per_station_actual,lstm_hidden_dim,batch_first=True)
        self.fc=nn.Linear(lstm_hidden_dim,pred_seq_len*target_dim); self.pred_seq_len,self.target_dim,self.num_stations=pred_seq_len,target_dim,num_stations_actual
    def forward(self,x_station_seq_batched):
        if x_station_seq_batched.shape[0]==0: return torch.empty((0,self.pred_seq_len,self.num_stations,self.target_dim),device=x_station_seq_batched.device)
        _,(hn,_)=self.lstm(x_station_seq_batched); last_hidden_state=hn[-1]; predictions_flat=self.fc(last_hidden_state)
        if x_station_seq_batched.shape[0]%self.num_stations!=0: raise ValueError(f"LSTM input batch {x_station_seq_batched.shape[0]} not div by num_stations {self.num_stations}")
        batch_size_inferred=x_station_seq_batched.shape[0]//self.num_stations
        return predictions_flat.view(batch_size_inferred,self.num_stations,self.pred_seq_len,self.target_dim).permute(0,2,1,3)

class UrbanEVDataset(Dataset):
    def __init__(self,X_dict_samples,Y_samples,graph_features_dict):
        self.X_dict_samples={key:torch.from_numpy(data_array).float() for key,data_array in X_dict_samples.items()}
        self.Y_samples=torch.from_numpy(Y_samples).float(); self.graph_features={}
        for key,val in graph_features_dict.items():
            if isinstance(val,np.ndarray): self.graph_features[key]=torch.from_numpy(val).float()
            elif isinstance(val,torch.Tensor): self.graph_features[key]=val.float() if val.dtype!=torch.long else val
            else: self.graph_features[key]=val
        self.sample_keys=list(self.X_dict_samples.keys()); self.num_samples=self.Y_samples.shape[0]
    def __len__(self): return self.num_samples
    def __getitem__(self,idx):
        if idx>=self.num_samples: raise IndexError(f"Index {idx} out of bounds for dataset with {self.num_samples} samples.")
        inputs_sample_specific={key:self.X_dict_samples[key][idx] for key in self.sample_keys}
        return {**inputs_sample_specific,**self.graph_features},self.Y_samples[idx]

def train_model(model,train_loader,optimizer,criterion,device,model_name="Model",actual_dims_dict=None):
    model.train(); total_loss,num_batches_processed_train=0,0
    for batch_inputs,batch_targets in train_loader:
        inputs_on_device={k:(v.to(device).float() if isinstance(v,torch.Tensor) and v.dtype!=torch.long else v.to(device)) for k,v in batch_inputs.items()}
        targets_on_device=batch_targets.to(device).float(); optimizer.zero_grad()
        if model_name=="LSTM_Baseline":
            hist_demand_raw_lstm,weather_hist_raw_lstm=inputs_on_device['hist_demand'],inputs_on_device['weather']
            B,T_hist,S,_=hist_demand_raw_lstm.shape
            D_hd_actual_lstm,D_w_actual_lstm=actual_dims_dict['hist_demand_dim'],actual_dims_dict['weather_dim']
            weather_expanded_for_lstm=weather_hist_raw_lstm.unsqueeze(2).repeat(1,1,S,1)
            lstm_input_features=torch.cat([hist_demand_raw_lstm,weather_expanded_for_lstm],dim=-1)
            lstm_input_reshaped=lstm_input_features.permute(0,2,1,3).reshape(B*S,T_hist,D_hd_actual_lstm+D_w_actual_lstm)
            outputs=model(lstm_input_reshaped)
        else: outputs=model(inputs_on_device)
        if outputs.shape!=targets_on_device.shape: print(f"Warn TRAIN: Shape mismatch {model_name}. Out:{outputs.shape}, Tgt:{targets_on_device.shape}. Skip batch."); continue
        loss=criterion(outputs,targets_on_device); loss.backward(); optimizer.step(); total_loss+=loss.item(); num_batches_processed_train+=1
    return total_loss/num_batches_processed_train if num_batches_processed_train>0 else 0

def evaluate_model(model,data_loader,criterion,device,model_name="Model",actual_dims_dict=None, shock_event_rt_cols=None, test_event_rt_df=None):
    if data_loader is None: return float('nan'),float('nan'),float('nan'), {},None,None
    if not hasattr(data_loader,'dataset') or len(data_loader.dataset)==0: return float('nan'),float('nan'),float('nan'), {},None,None
    try:
        temp_iter=iter(data_loader); first_batch_for_check=next(temp_iter,None)
        if first_batch_for_check is None: return float('nan'),float('nan'),float('nan'), {},None,None
        if not isinstance(first_batch_for_check,(list,tuple)) or len(first_batch_for_check)!=2: return float('nan'),float('nan'),float('nan'), {},None,None
        _fb_inputs,_fb_targets=first_batch_for_check
        if not isinstance(_fb_inputs,dict) or not isinstance(_fb_targets,torch.Tensor): return float('nan'),float('nan'),float('nan'), {},None,None
        del temp_iter,first_batch_for_check,_fb_inputs,_fb_targets
    except Exception as e_pre_loop: traceback.print_exc(); return float('nan'),float('nan'),float('nan'), {},None,None
    model.eval(); total_loss_eval,all_predictions_eval,all_targets_eval,num_batches_processed_eval=0,[],[],0
    preds_during_shock, targets_during_shock = {shock_col:[] for shock_col in shock_event_rt_cols or []}, {shock_col:[] for shock_col in shock_event_rt_cols or []}
    preds_normal, targets_normal = [], []
    current_sample_idx_in_test_df = 0
    with torch.no_grad():
        try:
            for batch_idx, batch_data_eval in enumerate(data_loader):
                if not isinstance(batch_data_eval,(list,tuple)) or len(batch_data_eval)!=2: continue
                batch_inputs_eval,batch_targets_eval=batch_data_eval
                if not isinstance(batch_inputs_eval,dict) or not isinstance(batch_targets_eval,torch.Tensor): continue
                if batch_targets_eval.shape[0]==0: continue
                inputs_on_device_eval={k:(v.to(device).float() if isinstance(v,torch.Tensor) and v.dtype!=torch.long else v.to(device)) for k,v in batch_inputs_eval.items()}
                targets_on_device_eval=batch_targets_eval.to(device).float()
                if model_name=="LSTM_Baseline":
                    hist_d_eval,weather_h_eval=inputs_on_device_eval['hist_demand'],inputs_on_device_eval['weather']
                    B,T,S,_=hist_d_eval.shape; D_hd,D_w=actual_dims_dict['hist_demand_dim'],actual_dims_dict['weather_dim']
                    weather_exp_eval=weather_h_eval.unsqueeze(2).repeat(1,1,S,1)
                    lstm_in_eval=torch.cat([hist_d_eval,weather_exp_eval],dim=-1).permute(0,2,1,3).reshape(B*S,T,D_hd+D_w)
                    outputs_eval=model(lstm_in_eval)
                else: outputs_eval=model(inputs_on_device_eval)
                if outputs_eval.shape[0]==0 and targets_on_device_eval.shape[0]==0: continue
                if outputs_eval.shape!=targets_on_device_eval.shape: print(f"Warn EVAL: Shape mismatch {model_name}. Out:{outputs_eval.shape}, Tgt:{targets_on_device_eval.shape}. Skip batch."); continue
                loss_eval=criterion(outputs_eval,targets_on_device_eval); total_loss_eval+=loss_eval.item()
                all_predictions_eval.append(outputs_eval.cpu()); all_targets_eval.append(targets_on_device_eval.cpu()); num_batches_processed_eval+=1
                if test_event_rt_df is not None and shock_event_rt_cols and not test_event_rt_df.empty:
                    batch_size_current = batch_targets_eval.shape[0]
                    for i in range(batch_size_current):
                        sample_abs_idx = current_sample_idx_in_test_df + i
                        if sample_abs_idx < len(test_event_rt_df):
                            is_any_shock_active = False
                            for shock_col_name in shock_event_rt_cols:
                                if shock_col_name in test_event_rt_df.columns and test_event_rt_df.iloc[sample_abs_idx][shock_col_name] > 0:
                                    preds_during_shock[shock_col_name].append(outputs_eval[i].cpu())
                                    targets_during_shock[shock_col_name].append(targets_on_device_eval[i].cpu())
                                    is_any_shock_active = True
                            if not is_any_shock_active: preds_normal.append(outputs_eval[i].cpu()); targets_normal.append(targets_on_device_eval[i].cpu())
                    current_sample_idx_in_test_df += batch_size_current
        except Exception as e_loop_eval: print(f"CRIT ERR loop {model_name} (eval): Exception: {e_loop_eval}"); traceback.print_exc(); return float('nan'),float('nan'),float('nan'), {},None,None
    if not all_predictions_eval or num_batches_processed_eval==0: print(f"[{model_name}] No valid batches processed in eval loop."); return float('nan'),float('nan'),float('nan'), {},None,None
    avg_loss_eval=total_loss_eval/num_batches_processed_eval; predictions_tensor_eval,targets_tensor_eval=torch.cat(all_predictions_eval,dim=0),torch.cat(all_targets_eval,dim=0)
    overall_mae=torch.mean(torch.abs(predictions_tensor_eval-targets_tensor_eval)).item(); overall_rmse=torch.sqrt(torch.mean((predictions_tensor_eval-targets_tensor_eval)**2)).item()
    shock_metrics = {}
    if test_event_rt_df is not None and shock_event_rt_cols and not test_event_rt_df.empty:
        if preds_normal and targets_normal:
            pn_t,tn_t=torch.stack(preds_normal),torch.stack(targets_normal)
            shock_metrics["normal_mae"]=torch.mean(torch.abs(pn_t-tn_t)).item(); shock_metrics["normal_rmse"]=torch.sqrt(torch.mean((pn_t-tn_t)**2)).item()
        for shock_col_name in shock_event_rt_cols:
            if preds_during_shock[shock_col_name] and targets_during_shock[shock_col_name]:
                ps_t,ts_t=torch.stack(preds_during_shock[shock_col_name]),torch.stack(targets_during_shock[shock_col_name])
                shock_metrics[f"{shock_col_name}_mae"]=torch.mean(torch.abs(ps_t-ts_t)).item(); shock_metrics[f"{shock_col_name}_rmse"]=torch.sqrt(torch.mean((ps_t-ts_t)**2)).item()
            else: shock_metrics[f"{shock_col_name}_mae"]=float('nan'); shock_metrics[f"{shock_col_name}_rmse"]=float('nan')
    return avg_loss_eval,overall_mae,overall_rmse,shock_metrics, predictions_tensor_eval, targets_tensor_eval

# VISUALIZATION FUNCTIONS
plt.style.use('seaborn-v0_8-whitegrid')
def save_plot_to_pdf(figure, filename_prefix="plot"):
    output_dir = "visualizations";
    if not os.path.exists(output_dir): os.makedirs(output_dir)
    filepath = os.path.join(output_dir, f"{filename_prefix}_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.pdf")
    try: figure.savefig(filepath, bbox_inches='tight'); print(f"Plot saved to {filepath}")
    except Exception as e: print(f"Error saving plot {filepath}: {e}")
    plt.close(figure)

def plot_overall_performance(results_dict, metrics_to_plot=['MAE', 'RMSE']):
    df_list = []
    for scenario, metrics_data in results_dict.items():
        for metric_name in metrics_to_plot:
            if metric_name in metrics_data: df_list.append({'Scenario': scenario, 'Metric': metric_name, 'Value': metrics_data[metric_name]})
    if not df_list: print("No data for overall performance plot."); return
    df_plot = pd.DataFrame(df_list)
    fig, ax = plt.subplots(figsize=(14, 8))
    sns.barplot(x='Scenario', y='Value', hue='Metric', data=df_plot, ax=ax, palette="pastel")
    ax.set_title('Overall Model Performance Comparison (Test Set)', fontsize=16)
    ax.set_ylabel('Metric Value', fontsize=12); ax.set_xlabel('Scenario', fontsize=12)
    plt.xticks(rotation=30, ha='right', fontsize=10); plt.yticks(fontsize=10)
    ax.legend(title='Metric', fontsize=10, title_fontsize=11)
    plt.tight_layout(); save_plot_to_pdf(fig, "overall_performance_comparison")

def plot_ablation_bar(results_dict, scenarios_to_compare, plot_title, metric='MAE', baseline_scenario_name=None):
    plot_data = []
    for scenario_name in scenarios_to_compare:
        if scenario_name in results_dict and metric in results_dict[scenario_name]:
            plot_data.append({'Scenario': scenario_name.replace("_AllEvents", "").replace("_TargetShocksExist",""), metric: results_dict[scenario_name][metric]})
    if not plot_data: print(f"No data for {plot_title} plot."); return
    df_plot = pd.DataFrame(plot_data)
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x='Scenario', y=metric, data=df_plot, ax=ax, palette="coolwarm")
    ax.set_title(plot_title, fontsize=16); ax.set_ylabel(metric, fontsize=12); ax.set_xlabel('')
    plt.xticks(rotation=20, ha='right', fontsize=10); plt.yticks(fontsize=10)
    if baseline_scenario_name and baseline_scenario_name in results_dict and metric in results_dict[baseline_scenario_name]:
        baseline_value = results_dict[baseline_scenario_name][metric]
        ax.axhline(baseline_value, ls='--', color='red', label=f'{baseline_scenario_name.replace("_AllEvents","")} ({metric}={baseline_value:.4f})')
        ax.legend(fontsize=10)
    plt.tight_layout(); save_plot_to_pdf(fig, plot_title.replace(" ", "_").lower())

def plot_dynamic_adaptation_curves(predictions_targets_dict, scenarios_to_compare, shock_column_name_in_event_rt, station_idx_to_plot=0, num_timesteps_to_plot=200, pred_step_to_plot=0):
    num_scenarios = len(scenarios_to_compare)
    if num_scenarios == 0: print("No scenarios to plot for dynamic adaptation."); return
    fig, axes = plt.subplots(num_scenarios, 1, figsize=(16, 4 * num_scenarios), sharex=True, squeeze=False)
    axes = axes.flatten()

    for i, scenario_name in enumerate(scenarios_to_compare):
        ax = axes[i]
        if scenario_name not in predictions_targets_dict or \
           predictions_targets_dict[scenario_name] is None or \
           'predictions' not in predictions_targets_dict[scenario_name] or \
           'targets' not in predictions_targets_dict[scenario_name]:
            ax.text(0.5, 0.5, f"Data not available for\n{scenario_name}", ha='center', va='center', fontsize=12, color='red')
            ax.set_title(f'Dynamic Adaptation: {scenario_name} (Data Missing)', fontsize=14)
            continue
        data = predictions_targets_dict[scenario_name]
        preds = data['predictions']; targets = data['targets']; event_rt_df_test = data.get('event_rt_df')
        if preds is None or targets is None: ax.text(0.5, 0.5, f"Preds/Targets missing for\n{scenario_name}", ha='center', va='center'); continue

        max_plot_idx = min(num_timesteps_to_plot, preds.shape[0])
        preds_station_step = preds[:max_plot_idx, pred_step_to_plot, station_idx_to_plot, 0]
        targets_station_step = targets[:max_plot_idx, pred_step_to_plot, station_idx_to_plot, 0]
        time_axis = np.arange(len(targets_station_step))
        ax.plot(time_axis, targets_station_step, label='Actual', color='black', linestyle='-',linewidth=1.5, alpha=0.8)
        ax.plot(time_axis, preds_station_step, label=f'Predicted', linestyle='--',linewidth=1.5, alpha=0.9)
        ax.set_title(f'Dynamic Adaptation: {scenario_name.replace("_AllEvents","")} (Zone {station_idx_to_plot}, T+{pred_step_to_plot+1})', fontsize=14)
        ax.set_ylabel('Scaled Value', fontsize=10); ax.tick_params(axis='both', which='major', labelsize=9)
        if event_rt_df_test is not None and shock_column_name_in_event_rt in event_rt_df_test.columns:
            shock_active = event_rt_df_test[shock_column_name_in_event_rt].iloc[:max_plot_idx].values > 0
            for t_idx_shock in time_axis[shock_active]:
                ax.axvspan(t_idx_shock - 0.5, t_idx_shock + 0.5, color='red', alpha=0.15, lw=0)
        ax.legend(fontsize=9); ax.grid(True, linestyle=':', alpha=0.6)
    fig.text(0.5, 0.01, 'Time Step (Sample Index in Test Set)', ha='center', fontsize=12)
    plt.tight_layout(rect=[0, 0.03, 1, 0.97]);
    save_plot_to_pdf(fig, f"dyn_adapt_shock_{shock_column_name_in_event_rt}_station{station_idx_to_plot}")

# --- Main Execution ---
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu"); print(f"Using device: {device}")
    criterion = nn.MSELoss() # DEFINED CRITERION
    if not PYG_AVAILABLE and STGA_GAT_OUT_CHANNELS_PER_HEAD > 0 : print("CRIT WARN: PyG not installed. STGA placeholder.")
    SIMULATED_SHOCK_EVENT_RT_COLS = ['shock_traffic_jam', 'shock_large_event']
    BASE_EVENT_RT_COLS = ['extreme_high_temp','extreme_low_temp','heavy_rain','is_weekend','is_morning_rush','is_evening_rush','is_night','is_summer','is_winter']
    ablation_scenarios = {
        "FullModel_AllEvents": {"name":"FullModel_AllEvents","use_poi":True,"use_weather_mafe":True,"disable_stga":False,"disable_dcf":False,"event_rt_column_names_to_use":BASE_EVENT_RT_COLS+SIMULATED_SHOCK_EVENT_RT_COLS},
        "StaticFusion_AllEvents": {"name":"StaticFusion_AllEvents","use_poi":True,"use_weather_mafe":True,"disable_stga":False,"disable_dcf":True,"event_rt_column_names_to_use":BASE_EVENT_RT_COLS+SIMULATED_SHOCK_EVENT_RT_COLS},
        "NoSTGA_AllEvents": {"name":"NoSTGA_AllEvents","use_poi":True,"use_weather_mafe":True,"disable_stga":True,"disable_dcf":False,"event_rt_column_names_to_use":BASE_EVENT_RT_COLS+SIMULATED_SHOCK_EVENT_RT_COLS},
        "NoPOI_AllEvents": {"name":"NoPOI_AllEvents","use_poi":False,"use_weather_mafe":True,"disable_stga":False,"disable_dcf":False,"event_rt_column_names_to_use":BASE_EVENT_RT_COLS+SIMULATED_SHOCK_EVENT_RT_COLS},
        "NoWeatherMAFE_AllEvents": {"name":"NoWeatherMAFE_AllEvents","use_poi":True,"use_weather_mafe":False,"disable_stga":False,"disable_dcf":False,"event_rt_column_names_to_use":BASE_EVENT_RT_COLS+SIMULATED_SHOCK_EVENT_RT_COLS},
        "EventRt_BaseOnly_TargetShocksExist": {"name":"EventRt_BaseOnly","use_poi":True,"use_weather_mafe":True,"disable_stga":False,"disable_dcf":False,"event_rt_column_names_to_use":BASE_EVENT_RT_COLS},
        "EventRt_ShockSignalsOnly_TargetShocksExist": {"name":"EventRt_ShockSignalsOnly","use_poi":True,"use_weather_mafe":True,"disable_stga":False,"disable_dcf":False,"event_rt_column_names_to_use":SIMULATED_SHOCK_EVENT_RT_COLS},
    }
    all_scenario_results = {}; all_predictions_targets_test = {}

    for scenario_name, current_ablation_config in ablation_scenarios.items():
        print(f"\n\n>>>>>>>> RUNNING SCENARIO: {scenario_name} <<<<<<<<")
        data_tuple = load_and_preprocess_urban_ev(DATA_PATH,HIST_SEQ_LEN,PRED_SEQ_LEN,TARGET_COLUMN,ablation_cfg=current_ablation_config)
        if data_tuple[0] is None: print(f"Data load fail for {scenario_name}. Skip."); continue
        (train_X_s,train_Y_s),(val_X_s,val_Y_s),(test_X_s,test_Y_s),graph_feat,tgt_scaler,actual_dims,df_event_rt_test_for_analysis = data_tuple
        if len(train_Y_s)==0 : print(f"CRIT ERR: Train data empty for {scenario_name}! Skip."); continue
        train_dataset=UrbanEVDataset(train_X_s,train_Y_s,graph_feat); val_dataset=UrbanEVDataset(val_X_s,val_Y_s,graph_feat) if len(val_Y_s)>0 else None
        test_dataset=UrbanEVDataset(test_X_s,test_Y_s,graph_feat) if len(test_Y_s)>0 else None
        train_loader=DataLoader(train_dataset,batch_size=BATCH_SIZE,shuffle=True,drop_last=True)
        val_loader=DataLoader(val_dataset,batch_size=BATCH_SIZE,shuffle=False) if val_dataset else None
        test_loader=DataLoader(test_dataset,batch_size=BATCH_SIZE,shuffle=False) if test_dataset else None
        print(f"--- DataLoader States for {scenario_name} ---")
        if train_dataset:print(f"Train dataset len:{len(train_dataset)}, Train loader batches:{len(train_loader) if train_loader else 0}")
        if val_dataset:print(f"Val dataset len:{len(val_dataset)}, Val loader batches:{len(val_loader) if val_loader else 0}")
        if test_dataset:print(f"Test dataset len:{len(test_dataset)}, Test loader batches:{len(test_loader) if test_loader else 0}")
        dmf_model_hyperparams = {'mafe_embed_dim':MAFE_EMBED_DIM,'stga_gat_out_channels_per_head':STGA_GAT_OUT_CHANNELS_PER_HEAD,
                                 'stga_gat_heads':STGA_GAT_HEADS,'stga_lstm_hidden_dim':STGA_LSTM_HIDDEN_DIM,
                                 'stga_out_dim':STGA_OUT_DIM,'cdyn_dim':CDYN_DIM,'fused_dim':FUSED_DIM,
                                 'pd_hidden_dim':PD_HIDDEN_DIM,'pred_seq_len':PRED_SEQ_LEN,'target_dim':TARGET_DIM}
        model_instance = DyMFusionChargeNet(dmf_model_hyperparams,actual_dims,ablation_cfg=current_ablation_config).to(device).float()
        print(f"\n--- Training model for scenario: {scenario_name} ---");
        optimizer=optim.Adam(model_instance.parameters(),lr=LR)
        scheduler=ReduceLROnPlateau(optimizer,mode='min',factor=0.5,patience=SCHEDULER_PATIENCE,min_lr=1e-7) # verbose removed
        best_val_loss_for_model=float('inf'); epochs_no_improve=0;
        for epoch in range(EPOCHS):
            train_loss_epoch=train_model(model_instance,train_loader,optimizer,criterion,device,model_name=scenario_name,actual_dims_dict=actual_dims)
            val_loss_epoch,val_mae_epoch,val_rmse_epoch, _, _, _ = float('nan'),float('nan'),float('nan'), {}, None, None
            if val_loader:
                val_loss_epoch,val_mae_epoch,val_rmse_epoch, _, _, _ = evaluate_model(model_instance,val_loader,criterion,device,model_name=scenario_name,actual_dims_dict=actual_dims)
                if not np.isnan(val_loss_epoch): scheduler.step(val_loss_epoch)
                if val_loss_epoch<best_val_loss_for_model: best_val_loss_for_model=val_loss_epoch; epochs_no_improve=0
                else: epochs_no_improve+=1
            if np.isnan(train_loss_epoch) or (val_loader and np.isnan(val_loss_epoch)): print(f"E {epoch+1} - {scenario_name} - NaN loss. Stop."); break
            current_lr=optimizer.param_groups[0]['lr']
            print(f"E {epoch+1}/{EPOCHS} - {scenario_name} - TrainL:{train_loss_epoch:.4f}, ValL:{val_loss_epoch:.4f}, ValMAE:{val_mae_epoch:.4f}, ValRMSE:{val_rmse_epoch:.4f}, LR:{current_lr:.7f}")
            if epochs_no_improve>=PATIENCE_EARLY_STOPPING: print(f"Early stop {scenario_name} E {epoch+1}."); break
        print(f"--- Evaluating {scenario_name} on Test Set ---");
        final_test_avg_loss,final_test_mae,final_test_rmse,test_shock_metrics, raw_preds, raw_tgts = float('nan'),float('nan'),float('nan'),{}, None, None
        current_shock_cols_for_analysis = [col for col in SIMULATED_SHOCK_EVENT_RT_COLS if col in actual_dims.get("event_rt_column_names",[])]
        if test_loader:
            final_test_avg_loss,final_test_mae,final_test_rmse,test_shock_metrics, raw_preds, raw_tgts = evaluate_model(
                model_instance,test_loader,criterion,device,model_name=scenario_name,actual_dims_dict=actual_dims,
                shock_event_rt_cols=current_shock_cols_for_analysis, test_event_rt_df=df_event_rt_test_for_analysis)
        all_scenario_results[scenario_name]={"MAE":final_test_mae,"RMSE":final_test_rmse,"AvgLoss":final_test_avg_loss, "ShockMetrics":test_shock_metrics}
        if raw_preds is not None and raw_tgts is not None:
            all_predictions_targets_test[scenario_name] = {"predictions":raw_preds.numpy(),"targets":raw_tgts.numpy(),"event_rt_df":df_event_rt_test_for_analysis.copy() if df_event_rt_test_for_analysis is not None else None}

    # LSTM Baseline
    print(f"\n\n>>>>>>>> RUNNING SCENARIO: LSTM_Baseline <<<<<<<<")
    data_tuple_lstm = load_and_preprocess_urban_ev(DATA_PATH,HIST_SEQ_LEN,PRED_SEQ_LEN,TARGET_COLUMN,ablation_cfg=DEFAULT_ABLATION_CONFIG)
    if data_tuple_lstm[0] is None: print(f"Data load fail for LSTM. Skip.");
    else:
        (train_X_lstm,train_Y_lstm),(val_X_lstm,val_Y_lstm),(test_X_lstm,test_Y_lstm),_,_,actual_dims_lstm,df_event_rt_lstm_test = data_tuple_lstm
        if len(train_Y_lstm)>0:
            graph_feat_lstm = data_tuple_lstm[3]
            train_ds_lstm=UrbanEVDataset(train_X_lstm,train_Y_lstm,graph_feat_lstm); val_ds_lstm=UrbanEVDataset(val_X_lstm,val_Y_lstm,graph_feat_lstm) if len(val_Y_lstm)>0 else None
            test_ds_lstm=UrbanEVDataset(test_X_lstm,test_Y_lstm,graph_feat_lstm) if len(test_Y_lstm)>0 else None
            train_load_lstm=DataLoader(train_ds_lstm,batch_size=BATCH_SIZE,shuffle=True,drop_last=True)
            val_load_lstm=DataLoader(val_ds_lstm,batch_size=BATCH_SIZE,shuffle=False) if val_ds_lstm else None
            test_load_lstm=DataLoader(test_ds_lstm,batch_size=BATCH_SIZE,shuffle=False) if test_ds_lstm else None
            lstm_model=LSTMBaseline(actual_dims_lstm['hist_demand_dim']+actual_dims_lstm['weather_dim'],64,PRED_SEQ_LEN,TARGET_DIM,actual_dims_lstm['num_zones']).to(device).float()
            optimizer_lstm=optim.Adam(lstm_model.parameters(),lr=LR)
            scheduler_lstm=ReduceLROnPlateau(optimizer_lstm,mode='min',factor=0.5,patience=SCHEDULER_PATIENCE,min_lr=1e-7) # verbose removed
            best_val_loss_lstm=float('inf'); epochs_no_improve_lstm=0;
            for epoch in range(EPOCHS):
                train_loss_lstm=train_model(lstm_model,train_load_lstm,optimizer_lstm,criterion,device,"LSTM_Baseline",actual_dims_lstm)
                val_loss_lstm,val_mae_lstm,val_rmse_lstm,_,_,_ = float('nan'),float('nan'),float('nan'),{},None,None
                if val_load_lstm:
                    val_loss_lstm,val_mae_lstm,val_rmse_lstm,_,_,_ = evaluate_model(lstm_model,val_load_lstm,criterion,device,"LSTM_Baseline",actual_dims_lstm)
                    if not np.isnan(val_loss_lstm): scheduler_lstm.step(val_loss_lstm)
                    if val_loss_lstm<best_val_loss_lstm: best_val_loss_lstm=val_loss_lstm; epochs_no_improve_lstm=0
                    else: epochs_no_improve_lstm+=1
                current_lr_lstm=optimizer_lstm.param_groups[0]['lr']
                print(f"E {epoch+1}/{EPOCHS} - LSTM_Baseline - TrainL:{train_loss_lstm:.4f}, ValL:{val_loss_lstm:.4f}, ValMAE:{val_mae_lstm:.4f}, ValRMSE:{val_rmse_lstm:.4f}, LR:{current_lr_lstm:.7f}")
                if epochs_no_improve_lstm>=PATIENCE_EARLY_STOPPING: print(f"Early stop LSTM E {epoch+1}."); break
            final_test_avg_loss_l,final_test_mae_l,final_test_rmse_l,lstm_shock_metrics, lstm_raw_preds, lstm_raw_tgts=float('nan'),float('nan'),float('nan'),{},None,None
            current_simulated_shock_cols_in_lstm_data = [col for col in SIMULATED_SHOCK_EVENT_RT_COLS if col in actual_dims_lstm.get("event_rt_column_names",[])]
            if test_load_lstm:
                 # df_event_rt_lstm_test was already correctly named from data_tuple_lstm
                 final_test_avg_loss_l,final_test_mae_l,final_test_rmse_l,lstm_shock_metrics, lstm_raw_preds, lstm_raw_tgts = evaluate_model(
                     lstm_model,test_load_lstm,criterion,device,"LSTM_Baseline",actual_dims_lstm,
                     shock_event_rt_cols=current_simulated_shock_cols_in_lstm_data,
                     test_event_rt_df=df_event_rt_lstm_test # Use the correctly named variable
                 )
            all_scenario_results["LSTM_Baseline"]={"MAE":final_test_mae_l,"RMSE":final_test_rmse_l,"AvgLoss":final_test_avg_loss_l, "ShockMetrics":lstm_shock_metrics}
            if lstm_raw_preds is not None and lstm_raw_tgts is not None:
                 all_predictions_targets_test["LSTM_Baseline"] = {"predictions":lstm_raw_preds.numpy(),"targets":lstm_raw_tgts.numpy(),"event_rt_df":df_event_rt_lstm_test.copy() if df_event_rt_lstm_test is not None else None} # Corrected

    print("\n\n--- ALL SCENARIO FINAL TEST RESULTS ---")
    for scenario_name, metrics in all_scenario_results.items():
        print(f"Scenario: {scenario_name:<45} MAE = {metrics['MAE']:.4f}, RMSE = {metrics['RMSE']:.4f}, AvgLoss = {metrics['AvgLoss']:.4f}")
        if "ShockMetrics" in metrics and metrics["ShockMetrics"]:
            print(f"  Shock Metrics for {scenario_name}:")
            for shock_type, val in metrics["ShockMetrics"].items():
                 if isinstance(val, float): print(f"    {shock_type:<30}: {val:.4f}")
                 else: print(f"    {shock_type:<30}: {val}")
        print("-" * 70)

    # --- VISUALIZATION SECTION ---
    print("\n\n--- Generating Visualizations ---")
    plot_overall_performance(all_scenario_results, metrics_to_plot=['MAE', 'RMSE'])
    stga_scenarios = ['FullModel_AllEvents', 'NoSTGA_AllEvents', 'LSTM_Baseline']
    stga_scenarios_present = [s for s in stga_scenarios if s in all_scenario_results]
    if len(stga_scenarios_present) > 1: plot_ablation_bar(all_scenario_results, stga_scenarios_present, "Impact of STGA (Spatiotemporal Modeling)", metric='MAE', baseline_scenario_name='LSTM_Baseline')
    dcf_scenarios = ['FullModel_AllEvents', 'StaticFusion_AllEvents']
    dcf_scenarios_present = [s for s in dcf_scenarios if s in all_scenario_results]
    if len(dcf_scenarios_present) == 2: plot_ablation_bar(all_scenario_results, dcf_scenarios_present, "Impact of DCF (Dynamic vs. Static Fusion)", metric='MAE')
    mafe_poi_scenarios = ['FullModel_AllEvents', 'NoPOI_AllEvents']; mafe_poi_scenarios_present = [s for s in mafe_poi_scenarios if s in all_scenario_results]
    if len(mafe_poi_scenarios_present) == 2: plot_ablation_bar(all_scenario_results, mafe_poi_scenarios_present,"Impact of POI Features (MAFE)", metric='MAE',baseline_scenario_name='NoPOI_AllEvents')
    mafe_weather_scenarios = ['FullModel_AllEvents', 'NoWeatherMAFE_AllEvents']; mafe_weather_scenarios_present = [s for s in mafe_weather_scenarios if s in all_scenario_results]
    if len(mafe_weather_scenarios_present) == 2: plot_ablation_bar(all_scenario_results, mafe_weather_scenarios_present,"Impact of Weather Features through MAFE", metric='MAE',baseline_scenario_name='NoWeatherMAFE_AllEvents')
    adaptation_scenarios = ['FullModel_AllEvents', 'StaticFusion_AllEvents', 'LSTM_Baseline']
    adaptation_scenarios_present = [s for s in adaptation_scenarios if s in all_predictions_targets_test]
    if len(adaptation_scenarios_present) > 0 and SIMULATED_SHOCK_EVENT_RT_COLS:
        plot_dynamic_adaptation_curves(all_predictions_targets_test, adaptation_scenarios_present,
                                       shock_column_name_in_event_rt=SIMULATED_SHOCK_EVENT_RT_COLS[0],
                                       station_idx_to_plot=0, num_timesteps_to_plot=150, pred_step_to_plot=0)
        if len(SIMULATED_SHOCK_EVENT_RT_COLS) > 1:
             plot_dynamic_adaptation_curves(all_predictions_targets_test, adaptation_scenarios_present,
                                       shock_column_name_in_event_rt=SIMULATED_SHOCK_EVENT_RT_COLS[1],
                                       station_idx_to_plot=0, num_timesteps_to_plot=150, pred_step_to_plot=0)
    shock_metric_data = []
    for scenario, metrics_data in all_scenario_results.items():
        if "ShockMetrics" in metrics_data and metrics_data["ShockMetrics"]:
            base_mae = metrics_data.get("MAE", float('nan')); normal_mae = metrics_data["ShockMetrics"].get("normal_mae", float('nan'))
            shock_metric_data.append({'Scenario': scenario, 'Period': 'Overall (Test)', 'MAE': base_mae})
            shock_metric_data.append({'Scenario': scenario, 'Period': 'Normal Periods', 'MAE': normal_mae})
            for shock_col in SIMULATED_SHOCK_EVENT_RT_COLS:
                shock_col_mae_key = f"{shock_col}_mae"
                if shock_col_mae_key in metrics_data["ShockMetrics"]:
                    shock_metric_data.append({'Scenario': scenario, 'Period': f"Shock: {shock_col.replace('shock_','')}", 'MAE': metrics_data["ShockMetrics"][shock_col_mae_key]})
    if shock_metric_data:
        df_shock_plot = pd.DataFrame(shock_metric_data)
        key_scenarios_for_shock_plot = [s for s in ['FullModel_AllEvents', 'StaticFusion_AllEvents', 'LSTM_Baseline'] if s in df_shock_plot['Scenario'].unique()]
        df_shock_plot_filtered = df_shock_plot[df_shock_plot['Scenario'].isin(key_scenarios_for_shock_plot)]
        if not df_shock_plot_filtered.empty:
            fig_shock, ax_shock = plt.subplots(figsize=(12, 7))
            sns.barplot(x='Scenario', y='MAE', hue='Period', data=df_shock_plot_filtered, ax=ax_shock, palette="Set2")
            ax_shock.set_title('Performance During Normal vs. Shock Periods (MAE)', fontsize=16); ax_shock.set_ylabel('MAE', fontsize=12)
            plt.xticks(rotation=15, ha='right'); plt.tight_layout(); save_plot_to_pdf(fig_shock, "shock_vs_normal_performance")
    print("Visualization generation complete. Check the 'visualizations' directory.")
    if not PYG_AVAILABLE and dmf_model_hyperparams.get('stga_gat_out_channels_per_head',0)>0: print("Reminder: STGA used placeholder GAT.")
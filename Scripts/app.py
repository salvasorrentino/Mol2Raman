import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import scipy
from Scripts.utils_model.utils_metrics import apply_mask
from Scripts.utils_model.utils_model import  nomi_file_in_cartella
from sklearn.metrics import confusion_matrix


Pred_names = nomi_file_in_cartella(r'.\data\predictions')


def display_molecule(selected_mol, prominence, up=False, upper=False, qm9s=False, upper_t=False):

    smile_target = selected_mol[r'smile']
    st.write(rf"Molecule's smile: " + smile_target)

    raman_pred = selected_mol['raman_pred'].tolist()
    raman_true = selected_mol['raman_true'].tolist()
    mask, _ = scipy.signal.find_peaks(raman_pred, prominence=prominence)
    mask = mask.tolist()
    raman_pred_mask = apply_mask(raman_pred, mask)

    fig_rnd_mol = go.Figure()

    # Implementation for rescaling the spectra
    mod = 0
    mod1 = 0
    mod2 = 0
    mod3 = 0
    if upper_t:
        # mod1 = 700
        # mod2 = 350
        mod3 = 500
    else:
        mod1 = 0
        mod2 = 350

    if up:
        mod = 1400
    elif upper:
        mod3 = 500
    elif qm9s:
        mod = 200
    else:
        mod = 0

    fig_rnd_mol.add_trace(go.Scatter(x=np.linspace((300 + mod + mod3), (2100 + mod + mod1), len(raman_true)), y=raman_true,
                                     mode='lines', name='True'))
    fig_rnd_mol.add_trace(go.Scatter(x=np.linspace((300 + mod + mod3), (2100 + mod + mod1), len(raman_true)), y=raman_pred_mask,
                                     mode='lines', name='Predicted_mask'))
    fig_rnd_mol.add_trace(go.Scatter(x=np.linspace((300 + mod + mod3), (2100 + mod + mod1), len(raman_true)), y=raman_pred,
                                     mode='lines', name='Predicted'))
    fig_rnd_mol.add_trace(go.Scatter(x=np.linspace((300 + mod + mod3), (2100 + mod + mod1), len(raman_true)),
                                     y=np.array(data['raman_true'].to_list()).mean(axis=0),
                                     mode='lines', name='Mean'))

    f1_prom = calc_gl_f1(selected_mol, prominence, tolerance=5)
    fig_rnd_mol.update_layout(title=selected_mol['smile'],
                              xaxis_title='Wavelength',
                              yaxis_title='Intensity')

    rmse_note = f'Relative RMSE: {selected_mol["rmse"]:.4f}'
    fig_rnd_mol.add_annotation(text=rmse_note,
                               x=0.5,
                               y=0.9,
                               showarrow=False,
                               font=dict(size=12, color='black'))

    st.plotly_chart(fig_rnd_mol)
    if "f1" in selected_mol.index:
        st.write(f'Relative F1 (prominence=0.3): {selected_mol["f1"]:.4f}')
        if f1_prom != selected_mol["f1"]:
            st.write(f'Relative F1 (prominence={prominence}): {f1_prom:.4f}')
    else:
        st.write(f'Relative F1 (prominence={prominence}): {f1_prom:.4f}')
    # Representation of the predicted spectra and the full real one
    fig_rnd_mol1 = go.Figure()
    if up:
        sel_mol = selected_mol['RAMAN_SPECTRUM_2']
    else:
        sel_mol = selected_mol['RAMAN_SPECTRUM_1']

    yy_pred = np.interp(np.linspace(0, 1, len(sel_mol)), np.linspace(0, 1,
                                            len(selected_mol['raman_pred'])), selected_mol['raman_pred'])

    mask, _ = scipy.signal.find_peaks(yy_pred, prominence=prominence)
    mask = mask.tolist()
    raman_pred_mask_int = apply_mask(yy_pred, mask)
    yy_true = np.interp(np.linspace(0, 1, len(sel_mol)), np.linspace(0, 1,
                                            len(sel_mol)), sel_mol)

    fig_rnd_mol1.add_trace(
        go.Scatter(x=np.linspace(300 + mod + mod3, 2100 + mod + mod1, len(yy_true)), y=yy_true, mode='lines',
                   name='True'))
    fig_rnd_mol1.add_trace(
        go.Scatter(x=np.linspace(300 + mod + mod3, 2100 + mod + mod1, len(yy_true)), y=raman_pred_mask_int, mode='lines',
                   name='Predicted'))

    fig_rnd_mol1.update_layout(title=selected_mol['smile'],
                              xaxis_title='Wavelength',
                              yaxis_title='Intensity')

    rmse_note = f'Relative RMSE: {selected_mol["rmse"]:.4f}'
    fig_rnd_mol1.add_annotation(text=rmse_note,
                               x=0.5,
                               y=0.9,
                               showarrow=False,
                               font=dict(size=12, color='black'))


    st.plotly_chart(fig_rnd_mol1)

    # Add a vertical line to the RMSE histogram
    fig_rmse.update_layout(annotations=[], shapes=[])
    fig_rmse.add_vline(x=selected_mol['rmse'], line_dash='dash', line_color='firebrick')
    st.plotly_chart(fig_rmse)

def plot_confusion_matrix(df):
    pred = (np.array(df.raman_pred.to_list()).reshape(-1) > 1).astype(int)
    true = (np.array(df.raman_true.to_list()).reshape(-1) > 1).astype(int)
    cm_matrix = confusion_matrix(pred, true)
    fig = plt.figure(figsize=(10, 4))
    sns.heatmap(cm_matrix/np.sum(cm_matrix), annot=True, fmt='.2%', cmap='Blues', )
    st.pyplot(fig)

def compute_rmse(pred, true):
    return np.sqrt(((np.array(pred) - np.array(true)) ** 2).sum() / 321)

@st.cache_data
def read_predictions(nome):
    preds = pd.read_parquet(r".\data\predictions/" + nome)
    #true = pd.read_pickle(r".\data\raw\test_dtf_data_smile_no_conv_inter_" + inter + ".pickle")
    preds['rmse'] = ''
    preds['rmse'] = preds.apply(lambda x: compute_rmse(x.raman_pred, x.raman_true), axis=1)
    return preds

opt = [x * 0.1 for x in range(0, 11)]
opt1 = [round(x, 1) for x in opt]
prominence = st.sidebar.selectbox("Select intensity prominence", options=opt1)
nome = st.sidebar.selectbox("Select prediction name", options=Pred_names)

if nome:
    up = False
    upper = False
    qm9s = False
    upper_t = False

    if '_UPPER_T' in nome:
        upper_t = True
    elif '_UPPER' in nome:
        upper = True
    elif '_UP' in nome:
        up = True
    if '_QM9S' in nome:
        qm9s = True
    data = read_predictions(nome)

    # Calculate RMSE histogram
    fig_rmse = px.histogram(data, x='rmse', nbins=500, title='RMSE Distribution')
    fig_rmse.update_layout(xaxis_range=[0, 20])
    config_name = nome.replace("pred_", "")
    config_name = config_name.replace(".parquet", "")
    config_model = pd.read_json(r".\models/" + config_name + r"/config/config.json")
    modification = config_model['description']['random_state']

    # Display the RMSE histogram
    st.plotly_chart(fig_rmse)
    st.write(f"Valid rmse: {data.rmse.mean()}")
    if 'f1' in data.columns:
        f1 = data['f1'].mean()
        st.write(f'Global F1: {f1}')
    st.write(f"Model modification: {modification}")

    # Add controls to the sidebar for molecule selection
    selection_method = st.sidebar.radio("Select molecule by:", ("Random", "RMSE Range", "SMILE", "CC(C#C)(C#N)C#N",
                                                                "CC1=NC[C@]2(CN2)CO1"))

    plot_confusion_matrix(data)

    if selection_method == "Random":
        button = st.sidebar.button("Select Random mol")
        if button:
            idx = np.random.randint(0, len(data), 1)[0]
            selected_mol = data.iloc[idx, :].copy()
            display_molecule(selected_mol, prominence, up, upper, qm9s, upper_t)

    if selection_method == "RMSE Range":
        min_rmse = st.sidebar.slider("Minimum RMSE", min_value=0.0, max_value=20.0, value=0.0)
        max_rmse = st.sidebar.slider("Maximum RMSE", min_value=0.0, max_value=20.0, value=1.0)

        filtered_data = data[(data['rmse'] >= min_rmse) & (data['rmse'] <= max_rmse)]

        if not filtered_data.empty:
            selected_mol = filtered_data.sample()
            display_molecule(selected_mol.iloc[0], prominence, up, upper, qm9s, upper_t)

    if selection_method == "SMILE":
        smile = st.sidebar.text_input("Enter SMILE", value="")
        smile_data = data[data['smile'] == smile]
        
        if not smile_data.empty:
            selected_mol = smile_data.sample()
            display_molecule(selected_mol.iloc[0], prominence, up, upper, qm9s, upper_t)

    if selection_method == "CC(C#C)(C#N)C#N":
        smile_data = data[data['smile'] == 'CC(C#C)(C#N)C#N']

        if not smile_data.empty:
            selected_mol = smile_data.sample()
            display_molecule(selected_mol.iloc[0], prominence, up, upper, qm9s, upper_t)

    if selection_method == "CC1=NC[C@]2(CN2)CO1":
        smile_data = data[data['smile'] == 'CC1=NC[C@]2(CN2)CO1']

        if not smile_data.empty:
            selected_mol = smile_data.sample()
            display_molecule(selected_mol.iloc[0], prominence, up, upper, qm9s, upper_t)

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from ipywidgets import FloatSlider, IntSlider, Checkbox, Label, VBox, GridBox, Layout, interactive_output, Button
from IPython.display import display

# --- Parámetros del sistema (modelo de planta de primer orden) ---
K = 1.0       # Ganancia
tau = 10.0    # Constante de tiempo del sistema

# --- Configuración inicial de la simulación ---
t = np.linspace(0, 100, 1000)
dt = t[1] - t[0]

# --- Función de simulación PID ---
def simulate_pid(Kp, Ki, Kd, T_ref, perturbation_start, perturbation_end, T_amb_perturb, fl_perturbacion, T_initial):
    T = np.zeros_like(t)        # Temperatura
    e = np.zeros_like(t)        # Error
    output = np.zeros_like(t)  # Señal de control

    P_term = np.zeros_like(t)
    I_term = np.zeros_like(t)
    D_term = np.zeros_like(t)

    integral = 0.0
    prev_error = 0.0

    T_amb_values = np.zeros_like(t)
    T_amb_base=T_initial

    if Kp==Ki==Kd==0:
        T_ref=T_initial

    if not fl_perturbacion:
        T_amb_perturb = T_amb_base
        perturbation_start = 0
        perturbation_end = 0        

    for i in range(len(t)):
        if perturbation_start <= t[i] <= perturbation_end:
            T_amb = T_amb_perturb
        else:
            T_amb = T_amb_base
        T_amb_values[i] = T_amb

        if i == 0:
            T[i] = T_initial
            e[i] = T_ref - T[i]
            continue

        e[i] = T_ref - T[i-1]

        P_term[i] = Kp * e[i]
        integral += e[i] * dt
        I_term[i] = Ki * integral
        derivativo = (e[i] - prev_error) / dt
        D_term[i] = Kd * derivativo

        output[i] = P_term[i] + I_term[i] + D_term[i]

        dTdt = (K * output[i] - (T[i-1] - T_amb)) / tau
        T[i] = T[i-1] + dTdt * dt

        prev_error = e[i]
    
    return T, P_term, I_term, D_term, output, T_amb_values, e

# --- Función para actualizar el gráfico ---
def update_plot(Kp, Ki, Kd, T_ref, perturbation_start, perturbation_end, T_amb_perturb, fl_perturbacion, T_initial, rango_error):
    if Kp==Ki==Kd==0:
        T_div=T_initial
        n=1
    else:
        T_div=T_ref
        n=3

    if T_amb_perturb==T_ref:
        fl_perturbacion=False
        
    T, P_term, I_term, D_term, output, T_amb_values, e = simulate_pid(Kp, Ki, Kd, T_ref, perturbation_start, perturbation_end, T_amb_perturb**n/T_div**(n-1), fl_perturbacion, T_initial)

    error_max=T_ref+rango_error
    error_min=T_ref-rango_error
    
    fig = make_subplots(rows=4, cols=1,
                        shared_xaxes=False, # Changed to False to allow individual x-axis titles
                        vertical_spacing=0.08,
                        subplot_titles=(
                            "Respuesta del sistema PID con perturbación",
                            "Componentes PID",
                            "Señal de control (output)",
                            "Error (e(t))"
                        ))

    fig.add_trace(go.Scatter(x=t, y=T, mode='lines', name='Temperatura (T)', line=dict(color='blue')), row=1, col=1)
    fig.add_trace(go.Scatter(x=t, y=[T_ref]*len(t), mode='lines', name='Valor nominal',
                             line=dict(dash='dash', color='red')), row=1, col=1)

    fig.add_trace(go.Scatter(x=t, y=[error_min]*len(t), mode='lines', name='Límite Inferior (18°C)',
                             line=dict(dash='dot', color='green')), row=1, col=1)
    fig.add_trace(go.Scatter(x=t, y=[error_max]*len(t), mode='lines', name='Límite Superior (26°C)',
                             line=dict(dash='dot', color='green')), row=1, col=1)
    lim_sup_y_g1=28
    lim_inf_y_g1=15
    
    fig.update_layout(yaxis1=dict(
        range=[lim_inf_y_g1, lim_sup_y_g1],
        tickvals=[15, 17, 18, 20, 22, 24, 25, 26, 28],
        ticktext=["15", "17", "18", "20", "22", "24", "25", "26", "28"]
    ))


    # Detectar franjas de falla (Temperatura por fuera del rango de error)
 
    franjas = []
    en_fr = False
    inicio = None
    
    for i in range(len(T)):
        if T[i] > error_max:
            if not en_fr:
                en_fr = True
                inicio = t[i]
        elif T[i] < error_min:
            if not en_fr:
                en_fr = True
                inicio = t[i]
        else:
            if en_fr:
                en_fr = False
                fin = t[i]
                franjas.append((inicio, fin))

    # Captura el último tramo si termina en la última posición
    if en_fr:
        franjas.append((inicio, t[-1]))
    
    # Agregar las franjas de falla
    for start, end in franjas:
        fig.add_shape(
            type="rect",
            xref="x1", yref="y1",  # paper en y para cubrir todo el eje Y
            x0=start, x1=end,
            y0=lim_inf_y_g1, y1=lim_sup_y_g1,
            fillcolor="rgba(255, 0, 0, 0.2)",  # rojo transparente
            line_width=0,
            layer="below"
        )
    
    fig.add_trace(go.Scatter(x=t, y=P_term, mode='lines', name='P (Proporcional)', line=dict(color='orange')), row=2, col=1)
    fig.add_trace(go.Scatter(x=t, y=I_term, mode='lines', name='I (Integral)', line=dict(color='green')), row=2, col=1)
    fig.add_trace(go.Scatter(x=t, y=D_term, mode='lines', name='D (Derivativo)', line=dict(color='purple')), row=2, col=1)

    fig.add_trace(go.Scatter(x=t, y=output, mode='lines', name='Señal de control (output)', line=dict(color='brown')), row=3, col=1)

    fig.add_trace(go.Scatter(x=t, y=e, mode='lines', name='Error', line=dict(color='magenta')), row=4, col=1)
    
    fig.update_layout(
        height=1000,
        margin=dict(l=50, r=50, t=80, b=50),
        hovermode="x unified",
        title_text="Simulación de Control PID de Temperatura",
        title_x=0.5
    )

    fig.update_yaxes(title_text="Temperatura (°C)", row=1, col=1)
    fig.update_yaxes(title_text="Componentes PID", row=2, col=1)
    fig.update_yaxes(title_text="Señal de control", row=3, col=1)
    fig.update_yaxes(title_text="Error", row=4, col=1) 
    
    # Set x-axis title for each subplot
    fig.update_xaxes(title_text="Tiempo (s)", row=4, col=1)

    fig.show()

# --- Crear los controles interactivos con ipywidgets ---

#Sintonizacion del controlador
Kp_inicial=2.0
Ki_inicial=5.0
Kd_inicial=1.0

Kp_slider = FloatSlider(min=0.0, max=10.0, step=0.1, value=Kp_inicial, description='Kp:')
Ki_slider = FloatSlider(min=0.0, max=10.0, step=0.1, value=Ki_inicial, description='Ki:')
Kd_slider = FloatSlider(min=0.0, max=9.0, step=0.1, value=Kd_inicial, description='Kd:')
T_ref_slider = FloatSlider(min=15.0, max=30.0, step=0.5, value=22.0, description='T_ref (°C):')

initial_perturb_start_value = 30
initial_perturb_end_value = 50

perturbation_start_slider = IntSlider(
    min=0, max=100, step=1, value=initial_perturb_start_value,
    description='Inicio Perturbación (s):', continuous_update=True
)
perturbation_end_slider = IntSlider(
    min=0, max=100, step=1, value=initial_perturb_end_value,
    description='Fin Perturbación (s):', continuous_update=True
)
T_amb_perturb_slider = FloatSlider(min=10.0, max=35.0, step=0.5, value=15.0, description='T_amb Perturbación (°C):')
chk_perturbacion = Checkbox(value=False, description='Perturbación', disabled=False, indent=False)
boton_reset_controlador = Button(description="Reiniciar PID", button_style="")

rango_error = IntSlider(min=0, max=10, step=1, value=4, description='Rango Error (+/-):', continuous_update=True)

# Función que se ejecutará al hacer clic
def reset_PID(b):
    Kp_slider.value=Kp_inicial
    Ki_slider.value=Ki_inicial
    Kd_slider.value=Kd_inicial

# Asignar la función al evento click
boton_reset_controlador.on_click(reset_PID)


T_initial_slider = FloatSlider(min=10.0, max=30.0, step=0.5, value=20.0, description='Temp. Inicial (°C):')

def on_perturbation_start_change(change):
    new_start_value = change['new']
    if perturbation_end_slider.value < new_start_value:
        perturbation_end_slider.value = new_start_value
    perturbation_end_slider.min = new_start_value

def on_perturbation_end_change(change):
    new_end_value = change['new']
    if perturbation_start_slider.value > new_end_value:
        perturbation_start_slider.value = new_end_value
    perturbation_start_slider.max = new_end_value

perturbation_start_slider.observe(on_perturbation_start_change, names='value')
perturbation_end_slider.observe(on_perturbation_end_change, names='value')

def toggle_perturbation_sliders(change):
    disable_sliders = not change['new']
    perturbation_start_slider.disabled = disable_sliders
    perturbation_end_slider.disabled = disable_sliders
    T_amb_perturb_slider.disabled = disable_sliders

chk_perturbacion.observe(toggle_perturbation_sliders, names='value')
toggle_perturbation_sliders({'new': chk_perturbacion.value})
on_perturbation_start_change({'new': perturbation_start_slider.value})
on_perturbation_end_change({'new': perturbation_end_slider.value})

c1 = VBox([Label("🎛️ Control PID"), Kp_slider, Ki_slider, Kd_slider, boton_reset_controlador])
c2 = VBox([Label("🌡️ Perturbación"), perturbation_start_slider, perturbation_end_slider, T_amb_perturb_slider, chk_perturbacion])
c3 = VBox([Label("⚙️ Configuraciones Adicionales"), T_initial_slider, T_ref_slider, rango_error])

layoutControles = GridBox(
    children=[c1, c2, c3],
    layout=Layout(
        grid_template_columns="repeat(3, auto)",
        grid_gap="10px",
        width="100%",
        justify_items="flex-start",
    )
)

interactive_plot = interactive_output(update_plot, {
    'Kp': Kp_slider,
    'Ki': Ki_slider,
    'Kd': Kd_slider,
    'T_ref': T_ref_slider,
    'perturbation_start': perturbation_start_slider,
    'perturbation_end': perturbation_end_slider,
    'T_amb_perturb': T_amb_perturb_slider,
    'fl_perturbacion': chk_perturbacion,
    'T_initial': T_initial_slider,
    'rango_error': rango_error
})

display(VBox([layoutControles, interactive_plot]))

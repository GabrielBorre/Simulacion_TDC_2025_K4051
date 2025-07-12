import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from ipywidgets import FloatSlider, IntSlider, Checkbox, Label, VBox, GridBox, Layout, interactive_output
from IPython.display import display

# --- Parámetros del sistema (modelo de planta de primer orden para humedad) ---
K_hum = 1.0        # Ganancia del sistema de humedad
tau_hum = 10.0     # Constante de tiempo del sistema de humedad
HR_amb_base = 60.0 # Humedad ambiente base (%)

# --- Configuración inicial de la simulación ---
t = np.linspace(0, 100, 1000)
dt = t[1] - t[0]

# --- Función de simulación del controlador Proporcional (P) para humedad ---
def simulate_proportional_humidity(Kp, HR_ref, HR_inicial, perturbation_start, perturbation_end, HR_amb_perturb, fl_perturbacion):
    HR = np.zeros_like(t)         # Humedad relativa
    e = np.zeros_like(t)          # Error
    output = np.zeros_like(t)     # Señal de control (salida del controlador P)

    P_term = np.zeros_like(t)     # Solo término Proporcional

    HR_amb_values = np.zeros_like(t)

    # Si la perturbación no está habilitada
    if not fl_perturbacion:
        HR_amb_perturb = HR_amb_base # La perturbación de humedad es la base
        # Set perturbation range to effectively zero duration
        perturbation_start = 0
        perturbation_end = 0       

    for i in range(len(t)):
        # Perturbación de humedad ambiente
        if perturbation_start <= t[i] <= perturbation_end:
            HR_amb = HR_amb_perturb
        else:
            HR_amb = HR_amb_base
        HR_amb_values[i] = HR_amb # Guardar HR_amb para graficar

        if i == 0:
            HR[i] = HR_inicial # Usamos HR_inicial para el primer punto
            e[i] = HR_ref - HR[i]
            continue

        e[i] = HR_ref - HR[i-1] # Error actual

        # Componente Proporcional (P)
        P_term[i] = Kp * e[i]
        
        # Señal de control total (solo P)
        output[i] = P_term[i]

        # Modelo de la planta (respuesta a la señal de control y perturbación en HR_amb)
        dHRdt = (K_hum * output[i] - (HR[i-1] - HR_amb)) / tau_hum
        HR[i] = HR[i-1] + dHRdt * dt
    
    return HR, P_term, output, HR_amb_values

# --- Función para actualizar el gráfico ---
def update_plot(Kp, HR_ref, HR_inicial, perturbation_start, perturbation_end, HR_amb_perturb, fl_perturbacion):
    HR, P_term, output, HR_amb_values = simulate_proportional_humidity(Kp, HR_ref, HR_inicial, perturbation_start, perturbation_end, HR_amb_perturb, fl_perturbacion)
    
    fig = make_subplots(rows=3, cols=1,
                        shared_xaxes=True,
                        vertical_spacing=0.08,
                        subplot_titles=(
                            "Respuesta del sistema de Humedad (Control Proporcional)",
                            "Señal de control (output)",
                            "Humedad ambiente (perturbada)"
                        ))

    # Subplot 1: Humedad Relativa
    fig.add_trace(go.Scatter(x=t, y=HR, mode='lines', name='Humedad Actual (HR)', line=dict(color='blue')), row=1, col=1)
    fig.add_trace(go.Scatter(x=t, y=[HR_ref]*len(t), mode='lines', name='Valor Nominal',
                             line=dict(dash='dash', color='red')), row=1, col=1)
    
    # Add acceptable range lines
    fig.add_trace(go.Scatter(x=t, y=[55]*len(t), mode='lines', name='Límite Superior Aceptable (55%)',
                             line=dict(dash='dot', color='green')), row=1, col=1) # Changed to dot for consistency
    fig.add_trace(go.Scatter(x=t, y=[45]*len(t), mode='lines', name='Límite Inferior Aceptable (45%)',
                             line=dict(dash='dot', color='green')), row=1, col=1) # Changed to dot for consistency

    fig.update_layout(yaxis1=dict(
        range=[30, 70],
        tickvals=[30, 35, 40, 45, 50, 55, 60, 65, 70],
        ticktext=["30", "35", "40", "45", "50", "55", "60", "65", "70"]
    ))

    # Subplot 2: Señal de control (output)
    fig.add_trace(go.Scatter(x=t, y=output, mode='lines', name='Señal de Control (P)', line=dict(color='brown')), row=2, col=1)

    # Subplot 3: Humedad ambiente (perturbada)
    fig.add_trace(go.Scatter(x=t, y=HR_amb_values, mode='lines', name='Humedad Ambiente (Perturbada)', line=dict(color='grey')), row=3, col=1)
    
    fig.update_layout(
        height=800,
        margin=dict(l=50, r=50, t=80, b=50),
        hovermode="x unified",
        title_text="Simulación de Control Proporcional (P) de Humedad",
        title_x=0.5
    )

    fig.update_yaxes(title_text="Humedad Relativa (%)", row=1, col=1)
    fig.update_yaxes(title_text="Señal de Control", row=2, col=1)
    fig.update_yaxes(title_text="HR_amb (%)", row=3, col=1)
    fig.update_xaxes(title_text="Tiempo (s)", row=3, col=1)

    fig.show()

# --- Crear los controles interactivos con ipywidgets ---
Kp_slider_hum = FloatSlider(min=0.0, max=10.0, step=0.1, value=2.0, description='Kp:')
HR_ref_slider = FloatSlider(min=30.0, max=70.0, step=1.0, value=50.0, description='HR_ref (%):')
HR_inicial_slider = FloatSlider(min=30.0, max=70.0, step=1.0, value=50.0, description='HR Inicial (%):') # Nuevo slider para HR inicial

# Define initial min/max for sliders to avoid immediate conflicts
initial_perturb_start_value_hum = 30
initial_perturb_end_value_hum = 50

perturbation_start_slider_hum = IntSlider(
    min=0, max=100, step=1, value=initial_perturb_start_value_hum,
    description='Inicio Perturbación (s):', continuous_update=True
)
perturbation_end_slider_hum = IntSlider(
    min=0, max=100, step=1, value=initial_perturb_end_value_hum,
    description='Fin Perturbación (s):', continuous_update=True
)
HR_amb_perturb_slider = FloatSlider(min=10.0, max=90.0, step=1.0, value=75.0, description='HR_amb Perturbación (%):')
chk_perturbacion_hum = Checkbox(value=False, description='Habilitar Perturbación', disabled=False, indent=False)


# --- Observadores para los sliders de perturbación (Lógica Corregida y Reforzada para Humedad) ---
def on_perturbation_start_change_hum(change):
    new_start_value = change['new']
    
    # Ensure end_slider's value is never less than new_start_value
    if perturbation_end_slider_hum.value < new_start_value:
        perturbation_end_slider_hum.value = new_start_value
    # Then set the min limit for the end slider
    perturbation_end_slider_hum.min = new_start_value


def on_perturbation_end_change_hum(change):
    new_end_value = change['new']

    # Ensure start_slider's value is never greater than new_end_value
    if perturbation_start_slider_hum.value > new_end_value:
        perturbation_start_slider_hum.value = new_end_value
    # Then set the max limit for the start slider
    perturbation_start_slider_hum.max = new_end_value


# Attach observers for humidity sliders
perturbation_start_slider_hum.observe(on_perturbation_start_change_hum, names='value')
perturbation_end_slider_hum.observe(on_perturbation_end_change_hum, names='value')


# --- Habilitar o deshabilitar sliders de perturbación para humedad ---
def toggle_perturbation_sliders_hum(change):
    disable_sliders = not change['new']
    perturbation_start_slider_hum.disabled = disable_sliders
    perturbation_end_slider_hum.disabled = disable_sliders
    HR_amb_perturb_slider.disabled = disable_sliders

    # IMPORTANT: When enabling, ensure the sliders are in a consistent state
    if not disable_sliders:
        # If start is greater than end, set end to start (or vice-versa)
        if perturbation_start_slider_hum.value > perturbation_end_slider_hum.value:
            perturbation_end_slider_hum.value = perturbation_start_slider_hum.value
        # Also ensure max/min bounds are correct after potential value changes
        # Triggering the observers with current values ensures bounds are synced
        on_perturbation_start_change_hum({'new': perturbation_start_slider_hum.value})
        on_perturbation_end_change_hum({'new': perturbation_end_slider_hum.value})


chk_perturbacion_hum.observe(toggle_perturbation_sliders_hum, names='value')

# Trigger initial state for perturbation sliders based on checkbox value
# Call this *after* setting up observers to ensure correct initial synchronization
toggle_perturbation_sliders_hum({'new': chk_perturbacion_hum.value})

# Manually trigger initial synchronization of slider bounds (redundant but safe)
# This handles the case where initial values might already be invalid BEFORE any user interaction
# if perturbation_start_slider_hum.value > perturbation_end_slider_hum.value:
#     perturbation_end_slider_hum.value = perturbation_start_slider_hum.value
on_perturbation_start_change_hum({'new': perturbation_start_slider_hum.value})
on_perturbation_end_change_hum({'new': perturbation_end_slider_hum.value})


# Columnas para los controles
c1_hum = VBox([Label("🎛️ Control Proporcional"), Kp_slider_hum])
c2_hum = VBox([Label("💧 Perturbación Humedad"), perturbation_start_slider_hum, perturbation_end_slider_hum, HR_amb_perturb_slider, chk_perturbacion_hum])
c3_hum = VBox([Label("⚙️ Configuraciones Adicionales"), HR_ref_slider, HR_inicial_slider])

# Layout para las columnas de los controles
layoutControles_hum = GridBox(
    children=[c1_hum, c2_hum, c3_hum],
    layout=Layout(
        grid_template_columns="repeat(3, auto)",
        grid_gap="10px",
        width="100%",
        justify_items="flex-start",
    )
)

interactive_plot_hum = interactive_output(update_plot, {
    'Kp': Kp_slider_hum,
    'HR_ref': HR_ref_slider,
    'HR_inicial': HR_inicial_slider,
    'perturbation_start': perturbation_start_slider_hum,
    'perturbation_end': perturbation_end_slider_hum,
    'HR_amb_perturb': HR_amb_perturb_slider,
    'fl_perturbacion': chk_perturbacion_hum
})

# Mostrar controles y gráfico juntos
display(VBox([layoutControles_hum, interactive_plot_hum]))



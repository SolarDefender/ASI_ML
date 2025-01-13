import streamlit as st
import requests
from datetime import datetime

api_url = "http://localhost:8000/predict"

st.title("Aplikacja do prognozowania zużycia energii elektrycznej")

st.markdown("""
Aplikacja umożliwia przewidywanie zużycia energii elektrycznej w gospodarstwie domowym na podstawie danych pogodowych i strefy docelowej. 
Dzięki temu użytkownicy mogą lepiej planować zużycie energii, optymalizować koszty oraz unikać przeciążeń w sieci energetycznej.
""")

current_datetime = datetime.now().strftime("%d/%m/%Y %H:%M")
st.write(f"Aktualna data i czas: {current_datetime}")

st.markdown("### Wprowadź dane pogodowe:")
temperature = st.number_input("Temperatura (°C)", value=6.559, help="Średnia temperatura w ciągu dnia w stopniach Celsjusza.")
humidity = st.slider("Wilgotność (%)", min_value=0, max_value=100, value=74, help="Poziom wilgotności powietrza w procentach (0–100%).")
wind_speed = st.number_input("Prędkość wiatru (m/s)", value=0.003, help="Średnia prędkość wiatru w metrach na sekundę.")
general_diffuse_flows = st.number_input("Całkowity rozproszony przepływ (kW)", value=0.051, help="Ilość energii rozproszonej w kilowatach.")
diffuse_flows = st.number_input("Rozproszony przepływ (kW)", value=0.119, help="Rozproszona energia promieniowania słonecznego w kilowatach.")

st.markdown("### Wybierz strefę docelową:")
target_zone = st.selectbox(
    "Strefa docelowa (1, 2, 3 lub pozostaw puste dla wszystkich stref):",
    [None, 1, 2, 3],
    help="Wybierz strefę, dla której chcesz uzyskać prognozę, lub pozostaw puste, aby uzyskać prognozy dla wszystkich stref."
)

if st.button("Utwórz prognozę"):
    payload = {
        "datetime": current_datetime,
        "temperature": temperature,
        "humidity": humidity,
        "wind_speed": wind_speed,
        "general_diffuse_flows": general_diffuse_flows,
        "diffuse_flows": diffuse_flows,
        "target_zone": target_zone
    }

    st.markdown("#### Wysyłane dane:")
    st.write(payload)

    try:
        response = requests.post(api_url, json=payload)
        response_data = response.json()

        if response.status_code == 200:
            if target_zone:
                st.success(f"Prognoza dla Strefy {target_zone}:")
                st.markdown(
                    f"""
                    <div style="border: 2px solid #28a745; border-radius: 10px; padding: 20px; margin: 10px 0; background-color: #e9f7ef;">
                        <h4 style="color: #155724;">Strefa {target_zone}</h4>
                        <p style="font-size: 18px; color: #28a745;"><b>Przewidywane zużycie:</b> {response_data['prediction'][0]} kW</p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            else:
                st.success("Prognoza dla wszystkich stref:")
                for idx, (zone, prediction) in enumerate(response_data["predictions"].items(), 1):
                    st.markdown(
                        f"""
                        <div style="border: 2px solid #007bff; border-radius: 10px; padding: 20px; margin: 10px 0; background-color: #e9ecef;">
                            <h4 style="color: #004085;">Strefa {idx}</h4>
                            <p style="font-size: 18px; color: #007bff;"><b>Przewidywane zużycie:</b> {prediction[0]} kW</p>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
        else:
            st.error(f"Błąd: {response_data.get('detail', 'Nieznany błąd')}")
    except Exception as e:
        st.error(f"Błąd połączenia z API: {e}")

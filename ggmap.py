import streamlit as st
import pydeck as pdk

# ตั้งค่าชื่อ WebApp
st.title("แอปแสดงตำแหน่งบนแผนที่ (หมุดสวยงาม)")

# รับค่า Latitude และ Longitude จากผู้ใช้
lat = st.number_input("กรอก Latitude", value=13.7563)  # ค่าเริ่มต้นเป็น กทม.
lon = st.number_input("กรอก Longitude", value=100.5018)

# URL ของไอคอนหมุด (สามารถใช้ URL รูปภาพของตัวเองได้)
ICON_URL = "https://upload.wikimedia.org/wikipedia/commons/e/ec/RedDot.svg"

# ฟังก์ชันเพิ่มไอคอนให้ pydeck
def add_icon_data(lat, lon):
    return [{
        "lat": lat,
        "lon": lon,
        "icon_data": {
            "url": ICON_URL,
            "width": 128,
            "height": 128,
            "anchorY": 128  # จุดอ้างอิงของไอคอน (ปรับให้พอดีกับตำแหน่ง)
        }
    }]

# ตรวจสอบค่าที่ป้อนเข้ามา
if st.button("แสดงตำแหน่ง"):
    # สร้างไอคอนเลเยอร์
    icon_layer = pdk.Layer(
        "IconLayer",
        data=add_icon_data(lat, lon),
        get_position=["lon", "lat"],
        get_icon="icon_data",
        get_size=4,  # ปรับขนาดไอคอน
        size_scale=10,  # ปรับขนาดเพิ่มเติม
        pickable=True,
    )

    # ตั้งค่าการมองเห็นแผนที่
    view_state = pdk.ViewState(
        latitude=lat,
        longitude=lon,
        zoom=15,
        pitch=0,
    )

    # แสดงแผนที่
    st.pydeck_chart(pdk.Deck(
        map_style="mapbox://styles/mapbox/streets-v11",
        initial_view_state=view_state,
        layers=[icon_layer],
    ))

    st.success(f"ตำแหน่งที่เลือก: Latitude {lat}, Longitude {lon}")

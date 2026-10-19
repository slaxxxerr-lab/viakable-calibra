"""Traducciones al inglés de la fase 7: nombre, guía rápida y reporte completo."""
EN = {
    "Descargar reporte completo (Excel)": "Download full report (Excel)", "Resumen": "Summary", "Pronóstico total": "Total forecast", "Plan": "Plan", "Indicador": "Indicator", "Valor": "Value",
    "Capacidad": "Capacity", "Notas y calidad de los datos": "Notes and data quality", "Pronóstico total {h} meses ({u})": "Total forecast {h} months ({u})", "Rango mínimo": "Minimum range",
    "Rango máximo": "Maximum range", "Precisión de validación %": "Validation accuracy %", "Ingreso estimado (MXN)": "Estimated revenue (MXN)", "Margen bruto (MXN)": "Gross margin (MXN)",
    "Margen bruto %": "Gross margin %", "Costo del metal (MXN)": "Metal cost (MXN)", "Inventario de seguridad ({u})": "Safety stock ({u})", "Datos hasta": "Data through",
    "Volumen ({u})": "Volume ({u})", "Ingreso (MXN)": "Revenue (MXN)", "Margen (MXN)": "Margin (MXN)", "Capacidad ({u}/mes)": "Capacity ({u}/month)",
    "Requerido máximo ({u}/mes)": "Peak required ({u}/month)", "Utilización máxima %": "Peak utilization %", "Meses sobre capacidad": "Months over capacity",
    "Crecimiento %": "Growth %", "Error de referencia %": "Reference error %",
    # guía rápida
    "Precios en vivo": "Live prices",
    "Aquí ves el cobre, el aluminio y el tipo de cambio. Se actualizan solos cada 15 minutos y todo el pronóstico, el plan y el margen se recalculan con ellos. Si falla internet, se usa el último precio guardado.":
        "Here you see copper, aluminum and the exchange rate. They refresh by themselves every 15 minutes and the whole forecast, plan and margin are recalculated with them. If the internet fails, the last saved price is used.",
    "Noticias y regulación": "News and regulation",
    "La app busca cada 30 minutos aranceles, T-MEC, normas y precios de metales, y los clasifica por relevancia. Tú decides qué hacer: ninguna noticia cambia el pronóstico sola.":
        "Every 30 minutes the app looks for tariffs, USMCA, regulations and metal prices, and ranks them by relevance. You decide what to do: no news item changes the forecast on its own.",
    "El modo simple muestra lo esencial. Cambia a avanzado para ver métodos, consenso, machine learning, señales, escenarios y permisos del equipo. En Personalizar cambias idioma, tema, color, tipografía, densidad y unidad. Puedes abrir esta guía otra vez cuando quieras.":
        "Simple mode shows the essentials. Switch to advanced to see methods, consensus, machine learning, signals, scenarios and team permissions. In Customize you change language, theme, color, typeface, density and unit. You can reopen this guide whenever you like.",
}
EN.update({"Sección": "Section", "Modo": "Mode"})
EN.update({"ver el pronóstico": "see the forecast"})
EN.update({"Animaciones": "Animations", "Si tu sistema pide reducir el movimiento, la app no anima aunque estén activadas.": "If your system asks to reduce motion, the app does not animate even when they are on."})
EN.update({"Modelos entrenados": "Models trained", "Precios": "Prices", "Noticias": "News", "guardado": "saved"})
EN.update({"Presentar": "Present", "Continuar": "Resume", "Pausar": "Pause", "Salir": "Exit", "Capital liberado": "Capital freed", "del inventario de seguridad": "of safety stock",
 "{x} contra la base": "{x} vs. base", "Capital liberado (MXN)": "Capital freed (MXN)", "Revisar Metales.": "Review Metals.",
 "Pronóstico actual": "Current forecast", "Precio del cobre en vivo": "Live copper price", "Una noticia regulatoria": "A regulatory news item",
 "Escenario: exención arancelaria del T-MEC": "Scenario: USMCA tariff exemption", "Anomalías y alertas": "Anomalies and alerts",
 "Este es el pronóstico de los próximos 6 meses con los datos de hoy.": "This is the forecast for the next 6 months with today's data.",
 "El cobre se mueve cada día. Aquí se simula una alza de 10%: el margen y el pronóstico se recalculan al instante.": "Copper moves every day. Here a 10% rise is simulated: margin and forecast are recalculated instantly.",
 "La app vigila aranceles y normas. Esta es la noticia más relevante guardada; tú decides qué hacer con ella.": "The app watches tariffs and regulations. This is the most relevant saved news item; you decide what to do with it.",
 "Qué pasaría si las exportaciones a EE. UU. quedaran exentas de arancel.": "What would happen if exports to the US were exempt from tariffs.",
 "Un pronóstico más preciso necesita menos inventario de seguridad, y eso libera capital de trabajo.": "A more accurate forecast needs less safety stock, which frees working capital.",
 "Lo que necesita atención hoy, ordenado por urgencia.": "What needs attention today, ranked by urgency.",
 "Cada factor suma o resta al pronóstico base. Aquí se ve el efecto de los dos escenarios.": "Each factor adds to or subtracts from the base forecast. Here you see the effect of the two scenarios.",
 "Esperamos {v} {u} en los próximos {h} meses, con {p}% de precisión de validación.": "We expect {v} {u} over the next {h} months, with {p}% validation accuracy.",
 "El cobre está en {x}{d}. Con +10%, el margen cambia {m} y el pronóstico {f} {u}.": "Copper is at {x}{d}. With +10%, margin changes {m} and the forecast {f} {u}.",
 "Con arancel de 0%, la exportación pasa de {a} a {b} {u} y el margen cambia {m}.": "With a 0% tariff, exports go from {a} to {b} {u} and margin changes {m}.",
 "Tus datos no traen ventas de exportación definidas, así que el arancel no mueve nada. Define la regla en Datos para ver este efecto.": "Your data has no export sales defined, so the tariff moves nothing. Define the rule in Data to see this effect.",
 "El inventario de seguridad libera {x} de capital y ahorra {y} al año en costo de mantener.": "Safety stock frees {x} of capital and saves {y} a year in holding cost.",
 "De {a} a {b} {u} con los dos escenarios.": "From {a} to {b} {u} with the two scenarios."})

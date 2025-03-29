from fastapi import FastAPI, Request
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime
import json
import random
import os
import logging
from typing import Dict, List, Any, Optional, Union
import logging
from fastapi import FastAPI, Request
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime
import json
import random
import os
from typing import Dict, List, Any, Optional, Union

# Rest 
# Настройка логирования
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = FastAPI(title="IT Salary Dashboard", 
              description="Дашборд для анализа зарплат IT-специалистов")

# Настройка шаблонов
templates = Jinja2Templates(directory="templates")

# Если есть папка static, подключаем её для статических файлов
if os.path.exists("static"):
    app.mount("/static", StaticFiles(directory="static"), name="static")


def load_data() -> pd.DataFrame:
    """
    Загрузка и предобработка данных из CSV файла.
    
    Returns:
        pd.DataFrame: Предобработанный DataFrame с данными о вакансиях
    """
    try:
        # Загрузка данных из нового CSV файла
        df = pd.read_csv("cleaned_file_6.csv", encoding="utf-8")
        
        # Базовая очистка данных
        df = df.dropna(subset=["Зарплата"])
        
        # Преобразование типов данных
        df["Зарплата"] = df["Зарплата"].astype(float)
        
        # Преобразование списков навыков из строкового представления в список
        df["Ключевые навыки"] = df["Ключевые навыки"].apply(
            lambda x: eval(x) if isinstance(x, str) else x
        )
        
        logger.info(f"Загружено {len(df)} записей из CSV файла")
        return df
    
    except Exception as e:
        logger.error(f"Ошибка при загрузке данных: {e}")
        # Возвращаем пустой DataFrame с нужными колонками в случае ошибки
        return pd.DataFrame(columns=["Год", "Месяц_текст", "Название должности", 
                                    "Уровень позиции", "Формат работы", 
                                    "Ключевые навыки", "Город", "Зарплата"])


def format_salary(value: float) -> str:
    """
    Форматирует значение зарплаты в удобный для отображения формат.
    
    Args:
        value (float): Значение зарплаты
        
    Returns:
        str: Отформатированное значение (например, "129k")
    """
    return f"{int(value/1000)}k"


def create_summary_table(df: pd.DataFrame) -> str:
    """
    Создает HTML-код для сводной таблицы профессий.
    
    Args:
        df (pd.DataFrame): DataFrame с данными
        
    Returns:
        str: HTML-код таблицы
    """
    # Группировка по должности и подсчет количества
    profession_counts = df["Название должности"].value_counts().reset_index()
    profession_counts.columns = ["Профессия", "Количество"]
    profession_counts["Доля от всех"] = (profession_counts["Количество"] / len(df) * 100).round(1)
    
    # Сортировка по количеству (убывание)
    profession_counts = profession_counts.sort_values("Количество", ascending=False)
    
    # Создание HTML для таблицы
    summary_table_html = ""
    for _, row in profession_counts.head(10).iterrows():
        summary_table_html += f"""
        <tr>
            <td>{row['Профессия']}</td>
            <td class="text-center">{row['Количество']}</td>
            <td class="text-end">{row['Доля от всех']}%</td>
        </tr>
        """
    
    return summary_table_html


def create_boxplot(df: pd.DataFrame) -> str:
    """
    Создает блочную диаграмму (boxplot) для зарплат по должностям.
    
    Args:
        df (pd.DataFrame): DataFrame с данными
        
    Returns:
        str: HTML-код графика
    """
    # Определяем топ-6 профессий по количеству
    top_professions = df["Название должности"].value_counts().nlargest(6).index.tolist()
    df_top = df[df["Название должности"].isin(top_professions)]
    
    # Создаем цветовую схему
    colors = px.colors.qualitative.Plotly
    
    # Создание box plot с улучшенным дизайном
    fig = px.box(
        df_top, 
        x="Название должности", 
        y="Зарплата",
        color="Название должности",
        color_discrete_sequence=colors,
        title="",
        labels={"Название должности": "Должность", "Зарплата": "Зарплата (руб.)"},
        points="all",  # Показываем все точки
        notched=False,  # Без выемки
        template="plotly_white"
    )
    
    # Настройка внешнего вида
    fig.update_layout(
        showlegend=False,
        height=500,
        margin=dict(l=20, r=20, t=30, b=20),
        font=dict(family="Segoe UI, Arial", size=12),
        xaxis=dict(
            title=dict(font=dict(size=14)),
            tickfont=dict(size=12)
        ),
        yaxis=dict(
            title=dict(font=dict(size=14)),
            tickfont=dict(size=12),
            gridcolor='rgba(0,0,0,0.05)'
        ),
        plot_bgcolor="white"
    )
    
    # Добавляем медианные значения в виде аннотаций
    for i, profession in enumerate(top_professions):
        median_value = df[df["Название должности"] == profession]["Зарплата"].median()
        fig.add_annotation(
            x=profession,
            y=median_value,
            text=f"{format_salary(median_value)}",
            showarrow=False,
            font=dict(size=12, color="black", family="Segoe UI, Arial"),
            bgcolor="rgba(255, 255, 255, 0.8)",
            bordercolor="rgba(0, 0, 0, 0.1)",
            borderwidth=1,
            borderpad=3,
            yshift=15
        )
    
    return fig.to_html(full_html=False, include_plotlyjs="cdn")


def create_trend_chart(df: pd.DataFrame) -> str:
    """
    Создает график тренда зарплат по месяцам и годам.
    
    Args:
        df (pd.DataFrame): DataFrame с данными
        
    Returns:
        str: HTML-код графика
    """
    # Создаем копию DataFrame для агрегации
    trend_df = df.copy()
    
    # Создаем комбинированную колонку для временного ряда
    trend_df['Период'] = trend_df['Год'].astype(str) + '-' + trend_df['Месяц_текст']
    
    # Определяем топ-5 должностей по количеству
    top_positions = trend_df["Название должности"].value_counts().nlargest(5).index.tolist()
    
    # Фильтруем данные по топ должностям
    filtered_df = trend_df[trend_df["Название должности"].isin(top_positions)]
    
    # Агрегируем данные по периодам и должностям
    agg_df = filtered_df.groupby(['Период', 'Название должности'])['Зарплата'].mean().reset_index()
    
    # Создаем линейный график по периодам
    fig = px.line(
        agg_df, 
        x="Период", 
        y="Зарплата", 
        color="Название должности",
        title="",
        labels={"Период": "Период", "Зарплата": "Средняя зарплата (руб.)"},
        template="plotly_white",
        markers=True  # Добавляем маркеры на линии
    )
    
    # Настройка внешнего вида
    fig.update_layout(
        height=500,
        margin=dict(l=20, r=20, t=30, b=20),
        font=dict(family="Segoe UI, Arial", size=12),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(size=12)
        ),
        xaxis=dict(
            title=dict(font=dict(size=14)),
            tickfont=dict(size=12)
        ),
        yaxis=dict(
            title=dict(font=dict(size=14)),
            tickfont=dict(size=12),
            gridcolor='rgba(0,0,0,0.05)'
        ),
        plot_bgcolor="white"
    )
    
    # Добавляем значения для последнего периода для каждой должности
    for position in top_positions:
        position_df = agg_df[agg_df["Название должности"] == position]
        if not position_df.empty:
            last_period = position_df["Период"].iloc[-1]
            last_value = position_df[position_df["Период"] == last_period]["Зарплата"].values[0]
            
            fig.add_annotation(
                x=last_period,
                y=last_value,
                text=f"{format_salary(last_value)}",
                showarrow=False,
                font=dict(size=11, color="black", family="Segoe UI, Arial"),
                bgcolor="rgba(255, 255, 255, 0.8)",
                bordercolor="rgba(0, 0, 0, 0.1)",
                borderwidth=1,
                borderpad=3,
                xshift=15
            )
    
    return fig.to_html(full_html=False, include_plotlyjs=False)


def create_salary_table(df: pd.DataFrame) -> str:
    """
    Создает HTML-код для детальной таблицы зарплат.
    
    Args:
        df (pd.DataFrame): DataFrame с данными
        
    Returns:
        str: HTML-код таблицы
    """
    # Берем случайную выборку из DataFrame для демонстрации
    # В реальном приложении здесь должна быть логика пагинации
    sample_df = df.sample(min(10, len(df)))
    
    # Создаем таблицу с детальными данными
    salary_table_html = ""
    for _, row in sample_df.iterrows():
        # Форматируем навыки для отображения
        skills = ", ".join(row["Ключевые навыки"]) if isinstance(row["Ключевые навыки"], list) else row["Ключевые навыки"]
        
        salary_table_html += f"""
        <tr>
            <td>{row['Название должности']}</td>
            <td>{row['Уровень позиции']}</td>
            <td>{row['Город']}</td>
            <td>{row['Год']}-{row['Месяц_текст']}</td>
            <td>{row['Формат работы']}</td>
            <td>{skills}</td>
            <td>{int(row['Зарплата']):,} ₽</td>
        </tr>
        """.replace(",", " ")
    
    return salary_table_html


def create_work_format_chart(df: pd.DataFrame) -> str:
    """
    Создает круговую диаграмму распределения по форматам работы.
    
    Args:
        df (pd.DataFrame): DataFrame с данными
        
    Returns:
        str: HTML-код графика
    """
    # Группируем данные по формату работы
    format_counts = df["Формат работы"].value_counts().reset_index()
    format_counts.columns = ["Формат работы", "Количество"]
    
    # Создаем круговую диаграмму
    fig = px.pie(
        format_counts, 
        values="Количество", 
        names="Формат работы",
        title="",
        template="plotly_white",
        hole=0.4,  # Создаем кольцевую диаграмму
        color_discrete_sequence=px.colors.qualitative.Pastel
    )
    
    # Настройка внешнего вида
    fig.update_layout(
        height=300,
        margin=dict(l=20, r=20, t=30, b=20),
        font=dict(family="Segoe UI, Arial", size=12),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.2,
            xanchor="center",
            x=0.5,
            font=dict(size=12)
        )
    )
    
    # Добавляем аннотации с процентами
    fig.update_traces(
        textposition='inside',
        textinfo='percent+label',
        textfont=dict(size=12, color="white")
    )
    
    return fig.to_html(full_html=False, include_plotlyjs=False)


def create_position_level_chart(df: pd.DataFrame) -> str:
    """
    Создает круговую диаграмму распределения по уровням позиций.
    
    Args:
        df (pd.DataFrame): DataFrame с данными
        
    Returns:
        str: HTML-код графика
    """
    # Группируем данные по уровню позиции
    level_counts = df["Уровень позиции"].value_counts().reset_index()
    level_counts.columns = ["Уровень позиции", "Количество"]
    
    # Создаем круговую диаграмму
    fig = px.pie(
        level_counts, 
        values="Количество", 
        names="Уровень позиции",
        title="",
        template="plotly_white",
        hole=0.4,  # Создаем кольцевую диаграмму
        color_discrete_sequence=px.colors.qualitative.Pastel
    )
    
    # Настройка внешнего вида
    fig.update_layout(
        height=300,
        margin=dict(l=20, r=20, t=30, b=20),
        font=dict(family="Segoe UI, Arial", size=12),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.2,
            xanchor="center",
            x=0.5,
            font=dict(size=12)
        )
    )
    
    # Добавляем аннотации с процентами
    fig.update_traces(
        textposition='inside',
        textinfo='percent+label',
        textfont=dict(size=12, color="white")
    )
    
    return fig.to_html(full_html=False, include_plotlyjs=False)


# def create_skills_heatmap(df: pd.DataFrame) -> str:
#     """
#     Создает тепловую карту востребованности навыков по должностям.
    
#     Args:
#         df (pd.DataFrame): DataFrame с данными
        
#     Returns:
#         str: HTML-код графика
#     """
#     # Получаем топ-10 должностей
#     top_positions = df["Название должности"].value_counts().nlargest(10).index.tolist()
    
#     # Получаем все уникальные навыки из списков
#     all_skills = set()
#     for skills_list in df["Ключевые навыки"]:
#         if isinstance(skills_list, list):
#             all_skills.update(skills_list)
    
#     # Получаем топ-15 самых популярных навыков
#     skill_counts = {}
#     for skills_list in df["Ключевые навыки"]:
#         if isinstance(skills_list, list):
#             for skill in skills_list:
#                 skill_counts[skill] = skill_counts.get(skill, 0) + 1
    
#     top_skills = sorted(skill_counts.items(), key=lambda x: x[1], reverse=True)[:15]
#     top_skills = [skill for skill, _ in top_skills]
    
#     # Создаем матрицу для тепловой карты
#     heatmap_data = []
#     for position in top_positions:
#         position_df = df[df["Название должности"] == position]
        
#         row_data = []
#         for skill in top_skills:
#             # Считаем, сколько раз навык встречается для данной должности
#             count = 0
#             for skills_list in position_df["Ключевые навыки"]:
#                 if isinstance(skills_list, list) and skill in skills_list:
#                     count += 1
            
#             # Нормализуем значение относительно количества вакансий для должности
#             normalized_value = count / len(position_df) if len(position_df) > 0 else 0
#             row_data.append(normalized_value)
        
#         heatmap_data.append(row_data)
    
#     # Создаем тепловую карту
#     fig = go.Figure(data=go.Heatmap(
#         z=heatmap_data,
#         x=top_skills,
#         y=top_positions,
#         colorscale='Viridis',
#         showscale=True,
#         colorbar=dict(
#             title="Частота навыка",
#             tickmode="array",
#             tickvals=[0, 0.5, 1],
#             ticktext=["Редко", "Средне", "Часто"],
#             ticks="outside"
#         )

#     ))
    
#     # Настройка внешнего вида
#     fig.update_layout(
#         height=400,
#         margin=dict(l=20, r=20, t=30, b=20),
#         font=dict(family="Segoe UI, Arial", size=12),
#         xaxis=dict(
#             title="",
#             tickangle=-45,
#             tickfont=dict(size=11)
#         ),
#         yaxis=dict(
#             title="",
#             tickfont=dict(size=11)
#         )
#     )
    
#     return fig.to_html(full_html=False, include_plotlyjs=False)

def create_skills_heatmap(df: pd.DataFrame) -> str:
    """
    Creates a heatmap showing the demand for skills across different positions.
    
    Args:
        df (pd.DataFrame): DataFrame with the data
        
    Returns:
        str: HTML code for the graph
    """
    # Get top-10 positions
    top_positions = df["Название должности"].value_counts().nlargest(10).index.tolist()
    
    # Get all unique skills from lists
    all_skills = set()
    for skills_list in df["Ключевые навыки"]:
        if isinstance(skills_list, list):
            all_skills.update(skills_list)
    
    # Get top-15 most popular skills
    skill_counts = {}
    for skills_list in df["Ключевые навыки"]:
        if isinstance(skills_list, list):
            for skill in skills_list:
                skill_counts[skill] = skill_counts.get(skill, 0) + 1
    
    top_skills = sorted(skill_counts.items(), key=lambda x: x[1], reverse=True)[:15]
    top_skills = [skill for skill, _ in top_skills]
    
    # Create matrix for heatmap
    heatmap_data = []
    for position in top_positions:
        position_df = df[df["Название должности"] == position]
        
        row_data = []
        for skill in top_skills:
            # Count how many times the skill appears for this position
            count = 0
            for skills_list in position_df["Ключевые навыки"]:
                if isinstance(skills_list, list) and skill in skills_list:
                    count += 1
            
            # Normalize value relative to the number of vacancies for the position
            normalized_value = count / len(position_df) if len(position_df) > 0 else 0
            row_data.append(normalized_value)
        
        heatmap_data.append(row_data)
    
    # Create heatmap with fixed colorbar configuration
    fig = go.Figure(data=go.Heatmap(
        z=heatmap_data,
        x=top_skills,
        y=top_positions,
        colorscale='Viridis',
        showscale=True,
        colorbar=dict(
            title=dict(
                text="Частота навыка",
                side="right"
            ),
            tickmode="array",
            tickvals=[0, 0.5, 1],
            ticktext=["Редко", "Средне", "Часто"],
            ticks="outside"
        )
    ))
    
    # Update layout
    fig.update_layout(
        height=400,
        margin=dict(l=20, r=20, t=30, b=20),
        font=dict(family="Segoe UI, Arial", size=12),
        xaxis=dict(
            title="",
            tickangle=-45,
            tickfont=dict(size=11)
        ),
        yaxis=dict(
            title="",
            tickfont=dict(size=11)
        )
    )
    
    return fig.to_html(full_html=False, include_plotlyjs=False)

def create_dashboard_data(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Создает все необходимые данные для дашборда.
    
    Args:
        df (pd.DataFrame): DataFrame с данными
        
    Returns:
        Dict[str, Any]: Словарь с данными для шаблона
    """
    try:
        # 1. Ключевые метрики
        median_salary = format_salary(df["Зарплата"].median())
        average_salary = format_salary(df["Зарплата"].mean())
        
        # 2. Сводная таблица профессий
        summary_table_html = create_summary_table(df)
        
        # 3. Распределение по форматам работы
        work_format_chart_html = create_work_format_chart(df)
        
        # 4. Распределение по уровням позиций
        position_level_chart_html = create_position_level_chart(df)
        
        # 5. Тепловая карта навыков
        skills_heatmap_html = create_skills_heatmap(df)
        
        # 6. Блочная диаграмма зарплат
        boxplot_html = create_boxplot(df)
        
        # 7. График зарплат по периодам
        trend_chart_html = create_trend_chart(df)
        
        # 8. Таблица зарплат
        salary_table_html = create_salary_table(df)
        
        # Собираем все данные для шаблона
        dashboard_data = {
            "median_salary": median_salary,
            "average_salary": average_salary,
            "summary_table": summary_table_html,
            "work_format_chart": work_format_chart_html,
            "position_level_chart": position_level_chart_html,
            "skills_heatmap": skills_heatmap_html,
            "boxplot_chart": boxplot_html,
            "trend_chart": trend_chart_html,
            "salary_table": salary_table_html
        }
        
        logger.info("Данные для дашборда успешно созданы")
        return dashboard_data
    
    except Exception as e:
        logger.error(f"Ошибка при создании данных для дашборда: {e}")
        # Возвращаем пустые данные в случае ошибки
        return {
            "median_salary": "0k",
            "average_salary": "0k",
            "summary_table": "<tr><td colspan='3'>Нет данных</td></tr>",
            "work_format_chart": "",
            "position_level_chart": "",
            "skills_heatmap": "",
            "boxplot_chart": "",
            "trend_chart": "",
            "salary_table": "<tr><td colspan='7'>Нет данных</td></tr>"
        }


@app.get("/", response_class=HTMLResponse)
async def dashboard(request: Request):
    """
    Основной маршрут для отображения дашборда.
    
    Args:
        request (Request): Объект запроса FastAPI
        
    Returns:
        TemplateResponse: Отрендеренный HTML-шаблон с данными
    """
    df = load_data()
    dashboard_data = create_dashboard_data(df)
    return templates.TemplateResponse("dashbord.html", {"request": request, **dashboard_data})


@app.get("/health", response_class=HTMLResponse)
async def health_check():
    """
    Эндпоинт для проверки работоспособности приложения.
    
    Returns:
        str: Статус приложения
    """
    return "OK"


if __name__ == "__main__":
    import uvicorn
    
    # Запуск сервера
    logger.info("Запуск сервера...")
    uvicorn.run(app, host="127.0.0.1", port=8000)

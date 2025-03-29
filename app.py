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
        # Загрузка данных из CSV файла
        df = pd.read_csv("cleaned_file_6.csv", encoding="utf-8")
        
        # Базовая очистка данных
        df = df.dropna(subset=["Зарплата"])
        
        # Преобразование типов данных
        df["Зарплата"] = df["Зарплата"].astype(float)
        
        # Более надежное преобразование навыков из строкового представления в список
        def parse_skills(skills_str):
            if pd.isna(skills_str):
                return []
            if isinstance(skills_str, list):
                return skills_str
                
            try:
                if isinstance(skills_str, str):
                    # Проверяем, что строка похожа на список
                    if skills_str.startswith('[') and skills_str.endswith(']'):
                        # Используем встроенный парсер вместо eval
                        import ast
                        return ast.literal_eval(skills_str)
                    else:
                        # Если это просто строка, разделяем по запятой
                        return [s.strip() for s in skills_str.split(',')]
            except Exception as e:
                print(f"Ошибка при парсинге навыков '{skills_str}': {e}")
                return []
            
            return []
        
        df["Ключевые навыки"] = df["Ключевые навыки"].apply(parse_skills)
        
        # Логирование информации о загруженных данных
        print(f"Загружено {len(df)} записей")
        print(f"Уникальных должностей: {df['Название должности'].nunique()}")
        print(f"Уникальных городов: {df['Город'].nunique()}")
        print(f"Диапазон зарплат: {df['Зарплата'].min()} - {df['Зарплата'].max()}")
        
        return df
    
    except Exception as e:
        print(f"Ошибка при загрузке данных: {e}")
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
    # Проверка на пустой DataFrame
    if df.empty:
        return "<tr><td colspan='3'>Нет данных</td></tr>"
    
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
    # Проверка на пустой DataFrame
    if df.empty:
        return "<div>Нет данных для отображения</div>"
    
    # Определяем топ-6 профессий по количеству
    profession_counts = df["Название должности"].value_counts()
    # Проверяем, что у нас есть достаточно данных
    if profession_counts.empty:
        return "<div>Недостаточно данных для построения графика</div>"
        
    # Берем только те профессии, которые встречаются не менее 3 раз
    valid_professions = profession_counts[profession_counts >= 3].index.tolist()
    
    # Если у нас менее 2 валидных профессий, показываем все профессии
    if len(valid_professions) < 2:
        valid_professions = profession_counts.index.tolist()
    
    # Ограничиваем до топ-6
    top_professions = valid_professions[:6]
    df_top = df[df["Название должности"].isin(top_professions)]
    
    # Проверяем, что у нас есть данные после фильтрации
    if df_top.empty:
        return "<div>Недостаточно данных для построения графика после фильтрации</div>"
    
    # Проверка наличия экстремальных выбросов и их ограничение при необходимости
    q1 = df_top["Зарплата"].quantile(0.25)
    q3 = df_top["Зарплата"].quantile(0.75)
    iqr = q3 - q1
    upper_bound = q3 + 3 * iqr  # Более мягкое ограничение (3*IQR вместо 1.5*IQR)
    
    df_top_filtered = df_top.copy()
    # Ограничиваем только очень экстремальные выбросы
    df_top_filtered.loc[df_top_filtered["Зарплата"] > upper_bound, "Зарплата"] = upper_bound
    
    # Создаем цветовую схему
    colors = px.colors.qualitative.Plotly
    
    # Создание box plot с улучшенным дизайном
    fig = px.box(
        df_top_filtered, 
        x="Название должности", 
        y="Зарплата",
        color="Название должности",
        color_discrete_sequence=colors,
        title="Распределение зарплат по должностям",
        labels={"Название должности": "Должность", "Зарплата": "Зарплата (руб.)"},
        points="all",  # Показываем все точки
        notched=False,  # Без выемки
        template="plotly_white"
    )
    
    # Настройка внешнего вида
    fig.update_layout(
        showlegend=False,
        height=500,
        margin=dict(l=40, r=40, t=60, b=40),  # Увеличены отступы
        font=dict(family="Segoe UI, Arial", size=12),
        xaxis=dict(
            title=dict(font=dict(size=14)),
            tickfont=dict(size=12),
            categoryorder='total descending'  # Сортировка категорий
        ),
        yaxis=dict(
            title=dict(font=dict(size=14)),
            tickfont=dict(size=12),
            gridcolor='rgba(0,0,0,0.05)'
        ),
        plot_bgcolor="white",
        autosize=True  # Важно для адаптивности
    )
    
    # Добавляем медианные значения в виде аннотаций
    for profession in top_professions:
        profession_data = df_top_filtered[df_top_filtered["Название должности"] == profession]
        if not profession_data.empty:
            median_value = profession_data["Зарплата"].median()
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
    
    boxplot_html = fig.to_html(full_html=False, include_plotlyjs=False, config={'responsive': True})
    
    return boxplot_html


def create_trend_chart(df: pd.DataFrame) -> str:
    """
    Creates an enhanced trend chart for salary dynamics.
    """
    # Проверка на пустой DataFrame
    if df.empty:
        return "<div>Нет данных для отображения</div>"
    
    # Prepare data
    trend_df = df.copy()
    trend_df['Период'] = trend_df['Год'].astype(str) + '-' + trend_df['Месяц_текст']
    
    # Get top positions (minimum 3 occurrences)
    position_counts = trend_df["Название должности"].value_counts()
    valid_positions = position_counts[position_counts >= 3].index.tolist()
    
    # If we have fewer than 2 valid positions, show all positions
    if len(valid_positions) < 2:
        valid_positions = position_counts.index.tolist()
    
    # Limit to top-5
    top_positions = valid_positions[:5]
    filtered_df = trend_df[trend_df["Название должности"].isin(top_positions)]
    
    # Check if we have data after filtering
    if filtered_df.empty:
        return "<div>Недостаточно данных для построения графика после фильтрации</div>"
    
    # Calculate aggregations by period and position
    agg_df = filtered_df.groupby(['Период', 'Название должности'])['Зарплата'].agg(
        ['mean', 'count']
    ).reset_index()
    agg_df.columns = ['Период', 'Название должности', 'Зарплата', 'Количество']
    
    # Check for periods with data
    if agg_df['Период'].nunique() < 2:
        return "<div>Недостаточно временных периодов для построения графика динамики (минимум 2 требуется)</div>"
    
    # Sort periods chronologically if they have the format Year-Month
    months_order = {'Январь': 1, 'Февраль': 2, 'Март': 3, 'Апрель': 4, 
                   'Май': 5, 'Июнь': 6, 'Июль': 7, 'Август': 8, 
                   'Сентябрь': 9, 'Октябрь': 10, 'Ноябрь': 11, 'Декабрь': 12}
    
    try:
        # Create a sorting key function
        def sort_key(period):
            year, month = period.split('-')
            return int(year) * 100 + months_order.get(month, 0)
        
        # Get unique periods and sort them
        unique_periods = sorted(agg_df['Период'].unique(), key=sort_key)
        
        # Create a mapping for ordering
        period_order = {period: i for i, period in enumerate(unique_periods)}
        
        # Apply the ordering to the DataFrame
        agg_df['period_order'] = agg_df['Период'].map(period_order)
        agg_df = agg_df.sort_values(['Название должности', 'period_order'])
        
        # Define ordered periods for each position
        positions_with_ordered_periods = {}
        for position in top_positions:
            position_periods = agg_df[agg_df['Название должности'] == position]['Период'].tolist()
            positions_with_ordered_periods[position] = position_periods
    except Exception as e:
        print(f"Ошибка при сортировке периодов: {e}")
        # If sorting fails, we'll use the data as is
    
    # Create enhanced line chart
    fig = go.Figure()
    
    colors = ['#2196F3', '#4CAF50', '#FFC107', '#9C27B0', '#FF5722']
    
    for idx, position in enumerate(top_positions):
        position_data = agg_df[agg_df["Название должности"] == position]
        
        # Skip if we don't have enough data points
        if len(position_data) < 2:
            continue
        
        fig.add_trace(go.Scatter(
            x=position_data["Период"],
            y=position_data["Зарплата"],
            name=position,
            line=dict(
                color=colors[idx % len(colors)],
                width=3,
                shape='spline'
            ),
            mode='lines+markers',
            marker=dict(
                size=8,
                symbol='circle'
            ),
            hovertemplate="<b>%{x}</b><br>" +
                         "Зарплата: %{y:,.0f}₽<br>" +
                         "Вакансий: %{text}<extra></extra>",
            text=position_data["Количество"]
        ))
    
    # Check if we've added any traces
    if len(fig.data) == 0:
        return "<div>Недостаточно данных для построения графика динамики зарплат</div>"
    
    fig.update_layout(
        title="Динамика зарплат по должностям",
        height=500,
        margin=dict(l=40, r=40, t=60, b=60),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.15,
            xanchor="center",
            x=0.5,
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor="rgba(0,0,0,0.1)",
            borderwidth=1
        ),
        xaxis=dict(
            title="",
            tickangle=-45,
            showgrid=True,
            gridcolor='rgba(0,0,0,0.1)'
        ),
        yaxis=dict(
            title="Средняя зарплата (₽)",
            showgrid=True,
            gridcolor='rgba(0,0,0,0.1)',
            tickformat=",d"
        ),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(family="Segoe UI, Arial", size=12),
        hovermode='x unified',
        autosize=True
    )
    
    trend_chart_html = fig.to_html(full_html=False, include_plotlyjs=False, config={'responsive': True})
    
    return trend_chart_html

def create_salary_table(df: pd.DataFrame) -> str:
    """
    Создает HTML-код для детальной таблицы зарплат.
    
    Args:
        df (pd.DataFrame): DataFrame с данными
        
    Returns:
        str: HTML-код таблицы
    """
    # Проверка на пустой DataFrame
    if df.empty:
        return "<tr><td colspan='7'>Нет данных</td></tr>"
    
    # Берем случайную выборку из DataFrame для демонстрации
    # В реальном приложении здесь должна быть логика пагинации
    sample_df = df.sample(min(10, len(df)))
    
    # Создаем таблицу с детальными данными
    salary_table_html = ""
    for _, row in sample_df.iterrows():
        # Форматируем навыки для отображения
        skills = ", ".join(row["Ключевые навыки"]) if isinstance(row["Ключевые навыки"], list) else row["Ключевые навыки"]
        
        # Обрезаем список навыков, если он слишком длинный
        if isinstance(skills, str) and len(skills) > 100:
            skills = skills[:97] + "..."
        
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
    Creates an enhanced donut chart for work format distribution.
    """
    # Проверка на пустой DataFrame
    if df.empty:
        return "<div>Нет данных для отображения</div>"
    
    # Define all possible work formats
    work_formats = ['гибрид', 'офис', 'удаленно', 'гибкий']
    
    # Count occurrences for each format and ensure all categories are included
    format_counts = df["Формат работы"].value_counts().reindex(work_formats, fill_value=0).reset_index()
    format_counts.columns = ["Формат работы", "Количество"]
    
    # Calculate percentages correctly based on total number of rows in DataFrame
    total = format_counts["Количество"].sum()  # Используем сумму только из форматов работы в format_counts
    if total == 0:
        return "<div>Нет данных для отображения</div>"
    
    format_counts["Процент"] = (format_counts["Количество"] / total * 100).round(1)
    
    # Create enhanced donut chart
    fig = go.Figure(data=[go.Pie(
        labels=format_counts["Формат работы"],
        values=format_counts["Количество"],
        hole=0.6,
        textinfo="label+percent",
        textposition="outside",
        textfont=dict(size=12, family="Segoe UI, Arial"),
        marker=dict(
            colors=['#4CAF50', '#2196F3', '#FFC107', '#9C27B0'],
            line=dict(color='white', width=2)
        ),
        hovertemplate="<b>%{label}</b><br>" +
                     "Количество: %{value}<br>" +
                     "Процент: %{percent}<extra></extra>"
    )])
    
    fig.update_layout(
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.15,  # Положение легенды ниже графика
            xanchor="center",
            x=0.5,
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor="rgba(0,0,0,0.1)",
            borderwidth=1
        ),
        margin=dict(l=20, r=20, t=20, b=60),  # Увеличен отступ снизу для легенды
        height=350,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(family="Segoe UI, Arial", size=12),
        autosize=True  # Важно для адаптивности
    )
    
    work_format_chart_html = fig.to_html(full_html=False, include_plotlyjs=False, config={'responsive': True})
    
    return work_format_chart_html


def create_position_level_chart(df: pd.DataFrame) -> str:
    """
    Creates an enhanced donut chart for position level distribution.
    """
    # Проверка на пустой DataFrame
    if df.empty:
        return "<div>Нет данных для отображения</div>"
    
    # Group data by position level
    level_counts = df["Уровень позиции"].value_counts().reset_index()
    level_counts.columns = ["Уровень позиции", "Количество"]
    
    # Calculate percentages
    total = level_counts["Количество"].sum()
    if total == 0:
        return "<div>Нет данных для отображения</div>"
    
    level_counts["Процент"] = (level_counts["Количество"] / total * 100).round(1)
    
    # Create enhanced donut chart
    fig = go.Figure(data=[go.Pie(
        labels=level_counts["Уровень позиции"],
        values=level_counts["Количество"],
        hole=0.6,
        textinfo="label+percent",
        textposition="outside",
        textfont=dict(size=12, family="Segoe UI, Arial"),
        marker=dict(
            colors=['#00BCD4', '#3F51B5', '#FF5722', '#9C27B0', '#4CAF50'],  # Больше цветов для разных уровней
            line=dict(color='white', width=2)
        ),
        hovertemplate="<b>%{label}</b><br>" +
                     "Количество: %{value}<br>" +
                     "Процент: %{percent}<extra></extra>"
    )])
    
    fig.update_layout(
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.15,  # Положение легенды ниже графика
            xanchor="center",
            x=0.5,
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor="rgba(0,0,0,0.1)",
            borderwidth=1
        ),
        margin=dict(l=20, r=20, t=20, b=60),  # Увеличен отступ снизу для легенды
        height=350,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(family="Segoe UI, Arial", size=12),
        autosize=True  # Важно для адаптивности
    )
    
    position_level_chart_html = fig.to_html(full_html=False, include_plotlyjs=False, config={'responsive': True})
    
    return position_level_chart_html


def create_skills_heatmap(df: pd.DataFrame) -> str:
    """
    Creates an enhanced heatmap for skills demand across positions.
    """
    # Проверка на пустой DataFrame
    if df.empty:
        return "<div>Нет данных для отображения</div>"
    
    # Get positions with at least 3 occurrences
    position_counts = df["Название должности"].value_counts()
    valid_positions = position_counts[position_counts >= 3].index.tolist()
    
    # If we have fewer than 2 valid positions, show all positions
    if len(valid_positions) < 2:
        valid_positions = position_counts.index.tolist()
    
    # Limit to top-8
    top_positions = valid_positions[:8]
    
    # Process skills
    # First, collect all skills and their counts
    all_skills = {}
    for idx, row in df.iterrows():
        position = row["Название должности"]
        if position not in top_positions:
            continue
            
        skills = row["Ключевые навыки"]
        if not isinstance(skills, list):
            continue
            
        for skill in skills:
            if not skill:  # Skip empty skills
                continue
            if skill not in all_skills:
                all_skills[skill] = 0
            all_skills[skill] += 1
    
    # Check if we have skills data
    if not all_skills:
        return "<div>Нет данных о навыках для отображения</div>"
    
    # Get top skills (at least 3 occurrences)
    top_skills = []
    for skill, count in sorted(all_skills.items(), key=lambda x: x[1], reverse=True):
        if count >= 3:  # Only include skills that appear at least 3 times
            top_skills.append(skill)
        if len(top_skills) >= 12:  # Limit to top 12 skills
            break
    
    # If we have fewer than 4 skills, include more
    if len(top_skills) < 4:
        top_skills = [skill for skill, _ in sorted(all_skills.items(), key=lambda x: x[1], reverse=True)[:12]]
    
    # Create heatmap matrix
    heatmap_data = []
    annotations = []
    
    for pos_idx, position in enumerate(top_positions):
        position_df = df[df["Название должности"] == position]
        row_data = []
        
        # Count valid position rows
        valid_rows = 0
        for _, pos_row in position_df.iterrows():
            if isinstance(pos_row["Ключевые навыки"], list):
                valid_rows += 1
        
        for skill_idx, skill in enumerate(top_skills):
            # Count occurrences of this skill for this position
            skill_count = 0
            for _, pos_row in position_df.iterrows():
                skills = pos_row["Ключевые навыки"]
                if isinstance(skills, list) and skill in skills:
                    skill_count += 1
            
            # Calculate percentage (avoid division by zero)
            percentage = skill_count / valid_rows if valid_rows > 0 else 0
            row_data.append(percentage)
            
            # Add percentage annotations
            annotations.append(
                dict(
                    x=skill_idx,
                    y=pos_idx,
                    text=f"{percentage*100:.0f}%" if percentage > 0 else "",
                    showarrow=False,
                    font=dict(
                        size=10,
                        color='white' if percentage > 0.3 else 'black'
                    )
                )
            )
        
        heatmap_data.append(row_data)
    
    # Convert to numpy array
    import numpy as np
    heatmap_data = np.array(heatmap_data)
    
    # Create heatmap
    fig = go.Figure(data=go.Heatmap(
        z=heatmap_data,
        x=top_skills,
        y=top_positions,
        colorscale='Viridis',
        showscale=True,
        colorbar=dict(
            title=dict(text="Частота навыка", side="right"),
            tickmode="array",
            tickvals=[0, 0.25, 0.5, 0.75, 1],
            ticktext=["0%", "25%", "50%", "75%", "100%"],
            ticks="outside",
            thickness=15
        )
    ))
    
    # Add annotations
    fig.update_layout(annotations=annotations)
    
    # Improve appearance
    fig.update_layout(
        title="Тепловая карта навыков по должностям",
        height=500,
        margin=dict(l=40, r=40, t=60, b=40),
        xaxis=dict(
            title="",
            tickangle=-45,
            tickfont=dict(size=12)
        ),
        yaxis=dict(
            title="",
            tickfont=dict(size=12)
        ),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(family="Segoe UI, Arial", size=12),
        autosize=True
    )
    
    skills_heatmap_html = fig.to_html(full_html=False, include_plotlyjs=False, config={'responsive': True})
    
    return skills_heatmap_html

def create_wordcloud_data(df: pd.DataFrame) -> List:
    """
    Создает данные для облака слов на основе навыков.
    
    Args:
        df (pd.DataFrame): DataFrame с данными
        
    Returns:
        List: Список словарей с данными для облака слов
    """
    # Проверка на пустой DataFrame
    if df.empty:
        return []
    
    # Подсчет частоты навыков
    skill_freq = {}
    for skills_list in df["Ключевые навыки"]:
        if isinstance(skills_list, list):
            for skill in skills_list:
                skill_freq[skill] = skill_freq.get(skill, 0) + 1
    
    # Преобразование в формат для облака слов
    wordcloud_data = [
        {"text": skill, "value": count}
        for skill, count in sorted(skill_freq.items(), key=lambda x: x[1], reverse=True)[:50]
    ]
    
    return wordcloud_data


def create_geo_chart(df: pd.DataFrame) -> str:
    """
    Creates a geographical bar chart for salary distribution by city.
    """
    # Проверка на пустой DataFrame
    if df.empty:
        return "<div>Нет данных для отображения</div>"
    
    # Group data by city
    city_data = df.groupby("Город")["Зарплата"].agg(
        ["mean", "median", "count"]
    ).reset_index()
    city_data.columns = ["Город", "Средняя_зарплата", "Медианная_зарплата", "Количество"]
    
    # Filter for cities with sufficient data
    city_data = city_data[city_data["Количество"] >= 5].sort_values("Средняя_зарплата", ascending=False).head(10)
    
    # Create horizontal bar chart
    fig = go.Figure()
    
    # Add bars for mean salary
    fig.add_trace(go.Bar(
        y=city_data["Город"],
        x=city_data["Средняя_зарплата"],
        name="Средняя зарплата",
        orientation="h",
        marker=dict(
            color="#2196F3",
            line=dict(width=1, color="#1565C0")
        ),
        hovertemplate="<b>%{y}</b><br>" +
                     "Средняя зарплата: %{x:,.0f}₽<br>" +
                     "Количество вакансий: %{text}<extra></extra>",
        text=city_data["Количество"]
    ))
    
    # Add markers for median salary
    fig.add_trace(go.Scatter(
        y=city_data["Город"],
        x=city_data["Медианная_зарплата"],
        name="Медианная зарплата",
        mode="markers",
        marker=dict(
            color="#FF5722",
            size=10,
            symbol="diamond",
            line=dict(width=2, color="#BF360C")
        ),
        hovertemplate="<b>%{y}</b><br>" +
                     "Медианная зарплата: %{x:,.0f}₽<extra></extra>"
    ))
    
    # Update layout
    fig.update_layout(
        height=500,
        margin=dict(l=20, r=20, t=20, b=20),
        xaxis=dict(
            title="Зарплата (₽)",
            showgrid=True,
            gridcolor='rgba(0,0,0,0.1)',
            tickformat=",d"
        ),
        yaxis=dict(
            title="",
            autorange="reversed"  # Для отображения городов сверху вниз по убыванию зарплаты
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(family="Segoe UI, Arial", size=12),
        autosize=True  # Важно для адаптивности
    )
    
    geo_chart_html = fig.to_html(full_html=False, include_plotlyjs=False, config={'responsive': True})
    
    return geo_chart_html


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """
    Главная страница дашборда с улучшенной обработкой ошибок.
    
    Args:
        request (Request): HTTP запрос
        
    Returns:
        HTMLResponse: Отрендеренный HTML шаблон
    """
    try:
        # Загрузка данных
        df = load_data()
        
        # Проверка, что у нас есть данные
        if df.empty:
            return templates.TemplateResponse(
                "error.html",
                {"request": request, "error_message": "Данные не найдены. Проверьте наличие файла cleaned_file_6.csv"}
            )
        
        # Базовая статистика для отладки
        print("=== Статистика данных ===")
        print(f"Общее количество записей: {len(df)}")
        print(f"Количество уникальных должностей: {df['Название должности'].nunique()}")
        print(f"Уникальные должности: {df['Название должности'].unique()}")
        print(f"Количество уникальных городов: {df['Город'].nunique()}")
        print(f"Уникальные уровни позиций: {df['Уровень позиции'].unique()}")
        print(f"Среднее/медиана зарплаты: {df['Зарплата'].mean():.2f} / {df['Зарплата'].median():.2f}")
        print(f"Мин/макс зарплаты: {df['Зарплата'].min():.2f} / {df['Зарплата'].max():.2f}")
        
        # Проверка типов данных ключевых колонок
        print("=== Типы данных колонок ===")
        print(df.dtypes)
        
        # Проверка наличия пропущенных значений
        print("=== Пропущенные значения ===")
        print(df.isna().sum())
        
        # Генерация компонентов дашборда
        try:
            stats = {
                'total_vacancies': len(df),
                'avg_salary': int(df['Зарплата'].mean()),
                'median_salary': int(df['Зарплата'].median()),
                'total_positions': df['Название должности'].nunique(),
                'total_cities': df['Город'].nunique(),
            }
        except Exception as e:
            print(f"Ошибка при расчете статистики: {e}")
            stats = {
                'total_vacancies': len(df),
                'avg_salary': 0,
                'median_salary': 0,
                'total_positions': 0,
                'total_cities': 0,
            }
        
        try:
            summary_table_html = create_summary_table(df)
        except Exception as e:
            print(f"Ошибка при создании сводной таблицы: {e}")
            summary_table_html = "<tr><td colspan='3'>Ошибка при создании сводной таблицы</td></tr>"
        
        try:
            boxplot_html = create_boxplot(df)
        except Exception as e:
            print(f"Ошибка при создании boxplot: {e}")
            boxplot_html = "<div>Ошибка при создании диаграммы распределения зарплат</div>"
        
        try:
            trend_chart_html = create_trend_chart(df)
        except Exception as e:
            print(f"Ошибка при создании графика тренда: {e}")
            trend_chart_html = "<div>Ошибка при создании графика динамики зарплат</div>"
        
        try:
            salary_table_html = create_salary_table(df)
        except Exception as e:
            print(f"Ошибка при создании таблицы зарплат: {e}")
            salary_table_html = "<tr><td colspan='7'>Ошибка при создании таблицы данных</td></tr>"
        
        try:
            work_format_chart_html = create_work_format_chart(df)
        except Exception as e:
            print(f"Ошибка при создании графика форматов работы: {e}")
            work_format_chart_html = "<div>Ошибка при создании диаграммы форматов работы</div>"
        
        try:
            position_level_chart_html = create_position_level_chart(df)
        except Exception as e:
            print(f"Ошибка при создании графика уровней позиций: {e}")
            position_level_chart_html = "<div>Ошибка при создании диаграммы уровней позиций</div>"
        
        try:
            skills_heatmap_html = create_skills_heatmap(df)
        except Exception as e:
            print(f"Ошибка при создании тепловой карты навыков: {e}")
            skills_heatmap_html = "<div>Ошибка при создании тепловой карты навыков</div>"
        
        try:
            geo_chart_html = create_geo_chart(df)
        except Exception as e:
            print(f"Ошибка при создании географического графика: {e}")
            geo_chart_html = "<div>Ошибка при создании географического графика</div>"
        
        try:
            wordcloud_data = create_wordcloud_data(df)
        except Exception as e:
            print(f"Ошибка при создании данных для облака слов: {e}")
            wordcloud_data = []
        
        # Рендеринг шаблона
        return templates.TemplateResponse(
            "dashbord.html",
            {
                "request": request,
                "stats": stats,
                "summary_table_html": summary_table_html,
                "boxplot_html": boxplot_html,
                "trend_chart_html": trend_chart_html,
                "salary_table_html": salary_table_html,
                "work_format_chart_html": work_format_chart_html,
                "position_level_chart_html": position_level_chart_html,
                "skills_heatmap_html": skills_heatmap_html,
                "geo_chart_html": geo_chart_html,
                "wordcloud_data": json.dumps(wordcloud_data),
            }
        )
        
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"Критическая ошибка при рендеринге главной страницы: {e}")
        print(error_details)
        return templates.TemplateResponse(
            "error.html",
            {"request": request, "error_message": f"Произошла ошибка: {str(e)}", "error_details": error_details}
        )

@app.get("/api/data", response_model=Dict[str, Any])
async def get_data():
    """
    API endpoint для получения данных.
    
    Returns:
        Dict[str, Any]: Словарь с данными
    """
    try:
        df = load_data()
        
        # Преобразуем DataFrame в список словарей
        data = df.head(100).to_dict(orient="records")
        
        return {
            "status": "success",
            "data": data,
            "total_records": len(df)
        }
        
    except Exception as e:
        logger.error(f"Ошибка при получении данных через API: {e}")
        return {
            "status": "error",
            "message": str(e)
        }


@app.get("/api/filters", response_model=Dict[str, List])
async def get_filters():
    """
    API endpoint для получения доступных фильтров.
    
    Returns:
        Dict[str, List]: Словарь с вариантами фильтров
    """
    try:
        df = load_data()
        
        filters = {
            "positions": sorted(df["Название должности"].unique().tolist()),
            "levels": sorted(df["Уровень позиции"].unique().tolist()),
            "cities": sorted(df["Город"].unique().tolist()),
            "work_formats": sorted(df["Формат работы"].unique().tolist()),
            "years": sorted(df["Год"].unique().tolist()),
            "months": sorted(df["Месяц_текст"].unique().tolist())
        }
        
        return {
            "status": "success",
            "filters": filters
        }
        
    except Exception as e:
        logger.error(f"Ошибка при получении фильтров через API: {e}")
        return {
            "status": "error",
            "message": str(e),
            "filters": {}
        }


if __name__ == "__main__":
    import uvicorn
    
    # Запуск сервера
    logger.info("Запуск сервера...")
    uvicorn.run(app, host="127.0.0.1", port=8000)
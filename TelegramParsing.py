# !python -m spacy download ru_core_news_sm
# !python -m spacy download ru_core_news_md
# !pip install rapidfuzz

import pandas as pd
import json
import re
import spacy
from rapidfuzz import fuzz

# Загрузка языковой модели
nlp = spacy.load("ru_core_news_sm")

# russian-cities.json из https://github.com/pensnarik/russian-cities/blob/master/russian-cities.json
with open("russian-cities.json", "r") as read_file:
    data = json.load(read_file)

RUSSIAN_CITIES = [d['name'] for d in data]
#У даляем, потому что парсер путает город с "белой" зарплатой
RUSSIAN_CITIES.remove("Белый")

def remove_duplicates(df, column, threshold=85):
    """
    Находит и удаляет дублирующиеся строки на основе схожести текста.

    Parameters:
        df (pd.DataFrame): DataFrame для обработки.
        column (str): Название столбца, в котором искать дубликаты.
        threshold (int): Порог схожести текста (по умолчанию 85).

    Returns:
        pd.DataFrame: DataFrame без дубликатов.
    """
    # Удаляем точные дубликаты
    df = df.drop_duplicates(subset=[column], keep='first').reset_index(drop=True)

    seen = []  # Список для уже обработанных текстов
    to_drop = []  # Индексы строк, которые нужно удалить

    for idx, text in enumerate(df[column]):
        for seen_text in seen:
            # Рассчитываем степень схожести текущей строки с уже обработанными
            similarity = fuzz.ratio(text, seen_text)
            if similarity >= threshold:  # Если схожесть превышает установленный порог
                to_drop.append(idx)  # Помечаем строку как дубликат
                break
        seen.append(text)  # Добавляем текущую строку в список обработанных

    # Удаляем строки, помеченные как дубликаты
    df = df.drop(index=to_drop).reset_index(drop=True)
    return df

def lemmatize_text(text):
    """
    Лемматизирует текст.

    Args:
        text (str): Входная строка текста.

    Returns:
        str: Лемматизированная строка текста.
    """
    doc = nlp(text)  # Обрабатываем текст с помощью spaCy
    lemmatized_tokens = [token.lemma_ for token in doc]  # Извлекаем леммы токенов
    return " ".join(lemmatized_tokens)

def preprocess_text(text):
    """
    Предобрабатывает текст вакансии: приводит к нижнему регистру, удаляет лишние символы.

    Args:
        text: Текст вакансии.

    Returns:
        str: Предобработанный текст.
    """
    # 1. Приведение к нижнему регистру
    text = text.lower()
    text = text.replace('\n', '.')
    text = text.replace('!', '.')
    text = text.replace('*', '')
    text = text.replace('"', '')

    # 2. Удаление HTML-тегов (если они есть)
    text = re.sub(r"<[^>]+>", "", text)

    # 3. Удаление лишних символов (кроме букв, цифр, пробелов и знаков препинания)
    text = re.sub(r"[^a-zA-Zа-яА-Я0-9\s.,?!-]+:", "", text)

    # 4. Замена нескольких пробелов одним
    text = re.sub(r"\s+", " ", text).strip()

    return text

def extract_job_title(text):
    """
    Извлекает название должности из текста вакансии.

    Args:
        text: Текст вакансии.

    Returns:
        str: Название должности или None, если не найдено.
    """
    text = preprocess_text(text)

    # Поиск по ключевым словам "приглашаем", "требуется", "ищем", "вакансия:" и т.д.
    title_match1 = re.search(r"(?:нужны|нужен|в поиске|в поисках|ищет(?! .)|ищу тебя,|на позицию|на работу|вакансия:|(?<!#)вакансия(?! от)|приглашаем|требуется|ищем?)\s+(.+?)\s+(?:в|на должность|в команду|в компанию|с опытом)", text, re.IGNORECASE)
    if title_match1:
        full_title = title_match1.group(1).strip()
        # Обрезаем до первой точки
        dot_index = full_title.find(".")
        if dot_index != -1:
            result = full_title[:dot_index].strip()
        else:
            result = full_title
        return result

    # Поиск с ключевым словом "Position:" (особенно полезно для англоязычных и смешанных текстов)
    title_match2 = re.search(r"Position:\s*(.+)", text, re.IGNORECASE)
    if title_match2:
        full_title = title_match2.group(1).strip()
        # Обрезаем до первой точки
        dot_index = full_title.find(".")
        if dot_index != -1:
            result = full_title[:dot_index].strip()
        else:
            result = full_title
        return result

    # Если ничего не найдено, попробуем вычленить первое упоминание должности
    if "программист" in text:
        return "программист"
    if "разработчик" in text:
        return "разработчик"
    if "аналитик" in text:
        return "аналитик"
    if "консультант" in text:
      return "консультант"
    if "менеджер" in text:
        return "менеджер"
    if "архитектор" in text:
        return "менеджер"
    if "руководитель" in text:
        return "руководитель"

    return None

def extract_position_level(text):
    """
    Извлекает уровень позиции (junior, middle, senior, lead, manager) из текста вакансии.

    Args:
        text: Текст вакансии.

    Returns:
        str: Уровень позиции или None, если не найдено.
    """
    text = lemmatize_text(text.lower())  # Для регистронезависимого поиска

    if re.search(r"\bjunior\b|\bмладший\b", text):
        return "junior"
    elif re.search(r"\bmiddle\b|\bспециалист\b", text):
        return "middle"
    elif re.search(r"\bsenior\b|\bстарший\b", text):
        return "senior"
    elif re.search(r"\blead\b|\bведущий\b", text):
        return "lead"
    elif re.search(r"\bmanager\b|\bруководитель\b", text):
        return "manager"
    elif re.search(r"\bфункциональный архитектор\b", text): #Дополнительное правило, часто подразумевает lead.
        return "lead"
    elif re.search(r"middle\s*\+\s*/\s*senior", text): #Добавлено правило "middle+/senior"
        return "middle/senior"
    return None

def extract_work_format(text):
    """
    Извлекает формат работы (удаленно, офис, гибрид, гибкий) из текста вакансии.

    Args:
        text: Текст вакансии.

    Returns:
        str: Формат работы или None, если не найдено.
    """
    text = text.lower()  # Для регистронезависимого поискаэ
    text = text.replace('ё', 'е')

    if re.search(r"\bгибкий\b|\bгибкая\b|\bудаленно или офис\b", text):
        return "гибкий" #или гибрид, или удаленно - нужно анализировать контекст
    elif re.search(r"\bгибрид\b|\bгибридный\b", text):
        return "гибрид"
    elif re.search(r"\bудаленно\b|\bудаленная работа\b|\bremote\b|\bудаленка\b|\bудаленный\b", text):
        return "удаленно"
    elif re.search(r"\bофис\b|\bв офисе\b", text):
        return "офис"
    elif re.search(r"\b100%\s+удаленно\b|\b100%\s+remote\b", text):
        return "удаленно"
    elif re.search(r"\bполностью\s+удаленно\b", text):
        return "удаленно"
    elif re.search(r"\bможно\s+удаленно\b", text):
        return "удаленно"
    return "офис"

def extract_skills(text):
    """
    Извлекает ключевые навыки (1С, SQL, Python, Java и т.д.) из текста вакансии.

    Args:
        text: Текст вакансии.

    Returns:
        list: Список ключевых навыков.
    """
    text = text.lower()
    skills = []

    if re.search(r"\b1c\b|\b1 c\b|\b1с\b|\b1 с\b", text):
        skills.append("1С")
    if re.search(r"\bsql\b", text):
        skills.append("SQL")
    if re.search(r"\bpython\b", text):
        skills.append("Python")
    if re.search(r"\bjava\b", text):
        skills.append("Java")
    if re.search(r"\bjavascript\b|\bjs\b|\bjavas cript\b", text):
        skills.append("JavaScript")
    if re.search(r"\bhtml\b", text):
        skills.append("HTML")
    if re.search(r"\bcss\b", text):
        skills.append("CSS")
    if re.search(r"\breact\b", text):
        skills.append("React")
    if re.search(r"\bangular\b", text):
        skills.append("Angular")
    if re.search(r"\bvue\.js\b|\bvuejs\b", text):
        skills.append("Vue.js")
    if re.search(r"\bnode\.js\b|\bnodejs\b", text):
        skills.append("Node.js")
    if re.search(r"\bdocker\b", text):
        skills.append("Docker")
    if re.search(r"\bkubernetes\b", text):
        skills.append("Kubernetes")
    if re.search(r"\baws\b", text):
        skills.append("AWS")
    if re.search(r"\bazure\b", text):
        skills.append("Azure")
    if re.search(r"\bgcp\b", text):
        skills.append("GCP")
    if re.search(r"\berp\b", text):
        skills.append("ERP")
    if re.search(r"\bупп\b", text):
        skills.append("УПП")
    if re.search(r"\bзуп\b", text):
        skills.append("ЗУП")
    if re.search(r"\bбп\b", text):
        skills.append("БП")
    if re.search(r"\bут\b", text):
        skills.append("УТ")
    if re.search(r"\bскд\b", text):
        skills.append("СКД")
    if re.search(r"\bxml\b", text):
        skills.append("XML")
    if re.search(r"\bjson\b", text):
        skills.append("JSON")
    if re.search(r"\brest\s*api\b", text):
        skills.append("REST API")  # REST API (с учетом пробелов)
    if re.search(r"\bsoap\b", text):
        skills.append("SOAP")
    if re.search(r"\badodb\b", text):
        skills.append("ADODB") # ADODB
    if re.search(r"\bdetarion\b", text):
        skills.append("Detarion") #Detarion
    if re.search(r"\bБСП\b", text):
        skills.append("БСП") #БСП
    return skills if skills else None

def extract_city(text):
    """Извлекает город с использованием регулярных выражений."""
    text = lemmatize_text(text)
    cities_str = '|'.join(RUSSIAN_CITIES)
    city_match = re.search(rf"\b({cities_str})\b", text, re.IGNORECASE)
    if city_match:
        return city_match.group(1).title()
    elif re.search(r"\bСПБ\b", text, re.IGNORECASE):
        return "Санкт-Петербург"
    elif re.search(r"\bМСК\b", text, re.IGNORECASE):
        return "Москва"
    # Выделение ключевых слов для определения географии
    official_keywords = ["ТК РФ", "ДМС", "белая", "официальное","официальная","штат"]
    # Ищем совпадение по ключевым словам
    if any(re.search(rf'\b{keyword}\b', text, re.IGNORECASE) for keyword in official_keywords):
        return "Россия"
    return "Мир"

def extract_salaries(text):
    """
    Извлекает числа до и после пробела вокруг дефиса, включая ровно 7 символов до и после,
    а также обрабатывает числа после слов "от" и "до".

    Parameters:
        text (str): Исходная строка.

    Returns:
        tuple: (salary_from, salary_to)
    """

    #Удалим временные диапазоны
    text = re.sub(r'\bс\s*\d{1,2}:\d{2}\s*до\s*\d{1,2}:\d{2}\b', '', text)  # "с 09:00 до 18:00"
    text = re.sub(r'\b\d{1,2}:\d{2}\s*-\s*\d{1,2}:\d{2}\b', '', text)  # "09:00-18:00"
    text = re.sub(r'\b\d{1,2}:\d{2}\s*до\s*\d{1,2}:\d{2}\b', '', text)  # "09:00 до 18:00"

    #Патерн
    salary_pattern = r'(\d[\d\s]*[кКККтыс\.]*\s*(?:-|–|до)\s*\d[\d\s]*[кКККтыс\.]*|от\s*\d[\d\s]*[кКтыс\.]*|до\s*\d[\d\s]*[кКтыс\.]*)'
    text = re.findall(salary_pattern, text, flags=re.IGNORECASE)

    #еще удаляем графики и номера телефонов и диапазаоны
    values_to_remove = ["8-17","08-17","09-18","9-18","10-19"]
    text = [item for item in text if item.strip() not in values_to_remove]
    #7 9
    text = [item for item in text if not (item.startswith('7 9') or item.startswith('+7 9') or item.startswith('79') or item.startswith('+79'))]
    #'4-5'
    text = [item for item in text if not re.match(r'^\d\s*[-–]\s*\d$', item.strip())]

    salary_from = None
    salary_to = None

    ### Этап 1: Поиск чисел вокруг дефиса
    if isinstance(text, list):
        text = " ".join(text)

    text = text.strip().lower().replace("к", "000").replace("тыс", "000").replace(" ", "").replace("  '", "")

    # Ищем дефис, перед которым и после которого минимум 3 цифры
    match = re.search(r'(\d[\d\s]*)\s*(?:-|–|до)\s*(\d[\d\s]*)', text)
    if match:
        # Берем до 7 символов до и после дефиса
        match_full = re.search(r'(.{0,7})\s*(?:-|–)\s*(.{0,7})', text)
        if match_full:
            before_dash = match_full.group(1).strip()
            after_dash = match_full.group(2).strip()
            # Извлекаем только числа
            salary_from = re.sub(r'\D', '', before_dash)
            salary_to = re.sub(r'\D', '', after_dash)

           # Увеличиваем, если числа в тысячах
            if salary_from and len(salary_from) == 3:
              salary_from = int(salary_from) * 1000
            if salary_to and len(salary_to) == 3:
              salary_to = int(salary_to) * 1000

          # Преобразуем в int, если есть значения
            salary_from = int(salary_from) if salary_from else None
            salary_to = int(salary_to) if salary_to else None

        # Проверяем, что числа не меньше 100
            if salary_from and salary_from < 100:
              salary_from = None
            if salary_to and salary_to < 100:
              salary_to = None

    #Этап 2: если поиск по - не дал результат.Поиск от и до
    if salary_from is None and salary_to is None:
      #проверка на оставшиеся диапазоны
      text = [item for item in text if not re.match(r'^\d{1,2}\s*[-–−―]\s*\d{1,2}$', item.strip())]
      # Преобразуем список обратно в строку
      text = " ".join(text).lower().replace(" ", "").replace(".", "").replace(",", "").replace("тыс", "000").replace("'", "")

      #Поиск слова "от" и извлечение чисел
      from_match = re.search(r'от(.{0,7})', text)
      if from_match:
            salary_from_match = re.search(r'\d{3,}', from_match.group(1))
            if salary_from_match:
                salary_str_from = salary_from_match.group()
                if len(salary_str_from ) >= 7:
                    salary_from = int(salary_str_from [:6])
                else:
                    salary_from = int(salary_str_from )

                if len(str(salary_str_from)) == 3:
                    salary_from  *= 1000

      #Поиск слова "до" и извлечение чисел
      to_match = re.search(r'до(.{0,7})', text)
      if to_match:
            salary_to_match = re.search(r'\d{3,}', to_match.group(1))
            if salary_to_match:
                salary_str_to = salary_to_match.group()
                if len(salary_str_to) >= 7:
                    salary_to = int(salary_str_to[:6])
                else:
                    salary_to = int(salary_str_to)

                if len(str(salary_to)) == 3:
                    salary_to *= 1000

    # Проверка значений
    if salary_from is not None and (int(float(salary_from)) < 10000 or int(float(salary_from)) > 5000000):
        salary_from = None
    if salary_to is not None and (int(float(salary_to)) < 10000 or int(float(salary_to)) > 5000000):
        salary_to = None

    if salary_from is not None and salary_to is not None and salary_from > salary_to:
        salary_from, salary_to = None, None

    return salary_from, salary_to

#заполнения NaN
def fill_na_salaries_with_mean(df):
    """
    Заполняет NaN значения в столбцах salary_from_real и salary_to_real

    Args:
        vacancies_df (pd.DataFrame): DataFrame с данными вакансий.

    Returns:
        pd.DataFrame: Обновленный DataFrame.
    """
    #Копируем DataFrame
    new_df = df.copy()


    ##Делаем проверку на корнер кейсы
    #Удаляем строки, где одновременно ['salary_from_real'] и ['salary_to_real'] равны NaN
    both_nan_filter = new_df['salary_from_real'].isna() & new_df['salary_to_real'].isna()
    new_df = new_df[~both_nan_filter]
    #удаляем строки где from > to (только где оба значения не NaN)
    invalid_range_filter = (new_df['salary_from_real'].notna() & new_df['salary_to_real'].notna() & (new_df['salary_from_real'] > new_df['salary_to_real'])
    )
    new_df = new_df[~invalid_range_filter]

    #Удаляем строки с зарплатами вне допустимого диапазона
    MIN_SALARY = 10000
    MAX_SALARY = 1000000
    invalid_salary = (
        (new_df['salary_from_real'] < MIN_SALARY) |
        (new_df['salary_from_real'] > MAX_SALARY) |
        (new_df['salary_to_real'].notna() & (new_df['salary_to_real'] < MIN_SALARY)) |
        (new_df['salary_to_real'].notna() & (new_df['salary_to_real'] > MAX_SALARY))
    )
    new_df = new_df[~invalid_salary]
    #заполняем NaN по алгоритму.
    #Если присутсвует только значение "до",то значение "от" будет равно "до" -25%.
    #Если присутствует только значение "от",то значение "до" будет равно "от"+50%

    #Заполняем NaN в колонке salary_from_real
    #for index, row in vacancies_new_df.iterrows():
        #if pd.isna(row['salary_from_real']):  # Если salary_from_real равно NaN
             #vacancies_new_df.loc[index, 'salary_from_real'] = int(row['salary_to_real'])*0.75


    #Заполняем NaN в колонке salary_to_real
    #for index, row in vacancies_new_df.iterrows():
        #if pd.isna(row['salary_to_real']):
          #vacancies_new_df.loc[index, 'salary_to_real'] = int(row['salary_from_real'])*1.5

    #2способ
    only_to = new_df['salary_from_real'].isna() & new_df['salary_to_real'].notna()
    only_from = new_df['salary_from_real'].notna() & new_df['salary_to_real'].isna()
    #Заполняем "от" = "до" * 0.75 (округление вниз)
    new_df.loc[only_to, 'salary_from_real'] = (new_df.loc[only_to, 'salary_to_real'] * 0.75).astype(int)
    #Заполняем "до" = "от" * 1.5 (округление вверх)
    new_df.loc[only_from, 'salary_to_real'] = (new_df.loc[only_from, 'salary_from_real'] * 1.5).astype(int)

    #Приводим к целым числам
    new_df['salary_from_real'] = new_df['salary_from_real'].astype(int)
    new_df['salary_to_real'] = new_df['salary_to_real'].astype(int)


    #проверка диапазонов и соотношения после заполнения
    invalid_from = new_df[(new_df['salary_from_real'] < MIN_SALARY)|(new_df['salary_from_real'] > MAX_SALARY)]
    invalid_to = new_df[(new_df['salary_to_real'] < MIN_SALARY)|(new_df['salary_to_real'] > MAX_SALARY)]
    invalid_range = new_df[new_df['salary_from_real'] > new_df['salary_to_real']]
    if not invalid_from.empty:
            print(f"⚠️ Некорректные 'от': {len(invalid_from)} строк (должны быть {MIN_SALARY}-{MAX_SALARY})")
    if not invalid_to.empty:
            print(f"⚠️ Некорректные 'до': {len(invalid_to)} строк (должны быть {MIN_SALARY}-{MAX_SALARY})")
    if not invalid_range.empty:
      print(f"⚠️ 'От' > 'до': {len(invalid_range)} строк")

    #проверяем, что больше не осталось NaN в колонках salary_from_real и salary_to_real
    nan_check = new_df[['salary_from_real', 'salary_to_real']].isna()
    if nan_check.any().any():
      nan_count = nan_check.sum()
      print(f"⚠️ Остались пропуски: from_real={nan_count['salary_from_real']}, to_real={nan_count['salary_to_real']}")

    #делаем столбец средней
    new_df['Зарплата'] = (new_df['salary_from_real'] + new_df['salary_to_real']) // 2

    return new_df

# Пример использования (замените "Vacans_1.csv" на имя вашего файла)
df = pd.read_csv("telegram_messages_1.csv")

# Словарь для преобразования числового месяца в текстовый
month_names = {
    1: "Январь",
    2: "Февраль",
    3: "Март",
    4: "Апрель",
    5: "Май",
    6: "Июнь",
    7: "Июль",
    8: "Август",
    9: "Сентябрь",
    10: "Октябрь",
    11: "Ноябрь",
    12: "Декабрь"
}

# Добавляем столбец с названием месяца
df['Месяц_текст'] = df['Месяц'].map(month_names)

df = remove_duplicates(df, "Вакансия", threshold=75)

# Применение функции извлечения названия должности к каждой вакансии
df["Название должности"] = df["Вакансия"].apply(extract_job_title)
# Применение функции извлечения уровня позиции к каждой вакансии
df["Уровень позиции"] = df["Вакансия"].apply(extract_position_level)
# Применение функции извлечения формата работы к каждой вакансии
df["Формат работы"] = df["Вакансия"].apply(extract_work_format)
# Применение функции извлечения ключевых навыков к каждой вакансии
df["Ключевые навыки"] = df["Вакансия"].apply(extract_skills)
# Применение функции извлечения города к каждой вакансии
df["Город"] = df["Вакансия"].apply(extract_city)
# Применение функции извлечения зарплаты к каждой вакансии
df["salary_from_real"], df["salary_to_real"] = zip(*df["Вакансия"].apply(extract_salaries))
df = fill_na_salaries_with_mean(df)

# Вычисляем среднюю зарплату
df['Зарплата'] = (df['salary_from_real'] + df['salary_to_real']) // 2

df = df.dropna(subset="Ключевые навыки")

# Сохраняем результаты в CSV с добавлением года и месяца в текстовом формате
output_df = df.copy()
# Переименовываем столбцы для соответствия требованиям выходного файла
output_columns = ["Год", "Месяц_текст", "Название должности", "Уровень позиции", "Формат работы", "Ключевые навыки", "Город", "Зарплата"]
output_df[output_columns].to_csv("output.csv", index=False)

print("Обработка завершена. Результаты сохранены в output.csv")

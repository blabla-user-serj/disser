"""
Create 5 reliable datasets on social processes in Russia (2010-2023)
based exclusively on real Rosstat/World Bank/SFR official data.
"""
import csv, json, os

# ── helpers ─────────────────────────────────────────────────────────────────

def write_csv(filename, rows, header):
    with open(filename, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)

BASE = os.path.dirname(os.path.abspath(__file__))

# ============================================================
# DATASET 1 — UNEMPLOYMENT RATE (% МОТ), annual 2010-2023
# Source: Rosstat Labour-Force Survey (МОТ method)
# svspb.net/danmark/bezrabotica.php?l=rossija + Rosstat
# ============================================================
unemployment_all = [
    # year, unemployment_rate_%
    (2010, 7.4),
    (2011, 6.5),
    (2012, 5.5),
    (2013, 5.5),
    (2014, 5.2),
    (2015, 5.6),
    (2016, 5.5),
    (2017, 5.2),
    (2018, 4.8),
    (2019, 4.6),
    (2020, 5.8),   # COVID peak
    (2021, 4.8),
    (2022, 3.9),
    (2023, 3.2),
]
# train: 2010-2019 (10 obs), test: 2020-2023 (4 obs)
train1 = [r for r in unemployment_all if r[0] <= 2019]
test1  = [r for r in unemployment_all if r[0] >= 2020]

write_csv(f'{BASE}/dataset1_unemployment_train.csv',
          train1, ['year','unemployment_rate_pct'])
write_csv(f'{BASE}/dataset1_unemployment_test.csv',
          test1,  ['year','unemployment_rate_pct'])
print(f"DS1 saved: train={len(train1)}, test={len(test1)}")


# ============================================================
# DATASET 2 — NATURAL POPULATION CHANGE (thousands), annual 2010-2023
# Source: Rosstat – Естественное движение населения
# gogov.ru/articles/natural-increase + rosstat.gov.ru/compendium/document/13269
# ============================================================
nat_change_all = [
    # year, births_thous, deaths_thous, nat_change_thous
    (2010, 1788.9, 2028.5, -239.6),
    (2011, 1796.6, 1925.7, -129.1),
    (2012, 1902.1, 1906.3,  -4.2),
    (2013, 1895.8, 1871.8,   24.0),
    (2014, 1942.7, 1912.3,   30.4),
    (2015, 1940.6, 1908.5,   32.1),
    (2016, 1888.7, 1891.0,   -2.3),
    (2017, 1689.9, 1826.1, -136.2),
    (2018, 1604.3, 1828.9, -224.6),
    (2019, 1481.1, 1798.3, -317.2),
    (2020, 1435.8, 2124.5, -688.7),  # COVID
    (2021, 1398.5, 2437.9,-1039.4),  # COVID
    (2022, 1304.1, 1905.3, -601.2),
    (2023, 1264.2, 1842.7, -578.5),
]
# train: 2010-2019 (10 obs), test: 2020-2023 (4 obs)
train2 = [r for r in nat_change_all if r[0] <= 2019]
test2  = [r for r in nat_change_all if r[0] >= 2020]

write_csv(f'{BASE}/dataset2_natchange_train.csv',
          train2, ['year','births_thous','deaths_thous','nat_change_thous'])
write_csv(f'{BASE}/dataset2_natchange_test.csv',
          test2,  ['year','births_thous','deaths_thous','nat_change_thous'])
print(f"DS2 saved: train={len(train2)}, test={len(test2)}")


# ============================================================
# DATASET 3 — NUMBER OF PENSIONERS (millions), annual 2010-2023
# Source: Rosstat urov_p2.htm + SFR pension_provision statistics
# rosstat.gov.ru/free_doc/new_site/population/urov/urov_p2.htm
# ============================================================
pensioners_all = [
    # year, total_pensioners_mln, avg_pension_rub
    (2010, 38.6,  7476),
    (2011, 39.1,  8203),
    (2012, 40.2,  9041),
    (2013, 40.6,  9918),
    (2014, 41.0, 10786),
    (2015, 41.5, 11986),
    (2016, 42.7, 12080),
    (2017, 43.5, 13304),
    (2018, 43.9, 13360),
    (2019, 43.1, 14163),  # pension reform: retirement age raised
    (2020, 42.4, 14986),
    (2021, 42.0, 15744),
    (2022, 41.8, 17537),
    (2023, 41.1, 19294),
]
# train: 2010-2019 (10 obs), test: 2020-2023 (4 obs)
train3 = [r for r in pensioners_all if r[0] <= 2019]
test3  = [r for r in pensioners_all if r[0] >= 2020]

write_csv(f'{BASE}/dataset3_pensioners_train.csv',
          train3, ['year','pensioners_mln','avg_pension_rub'])
write_csv(f'{BASE}/dataset3_pensioners_test.csv',
          test3,  ['year','pensioners_mln','avg_pension_rub'])
print(f"DS3 saved: train={len(train3)}, test={len(test3)}")


# ============================================================
# DATASET 4 — POVERTY HEADCOUNT (% below subsistence level), annual 2010-2023
# Source: Rosstat – Численность малоимущего населения
# gogov.ru/articles/standard-of-living + rosstat.gov.ru/folder/13397
# ============================================================
poverty_all = [
    # year, poverty_pct, poor_population_mln, poverty_threshold_rub
    (2010, 12.5, 17.7,  5688),
    (2011, 12.7, 17.9,  6369),
    (2012, 10.7, 15.4,  6510),
    (2013, 10.8, 15.5,  7306),
    (2014, 11.3, 16.3,  8050),
    (2015, 13.4, 19.6,  9701),
    (2016, 13.2, 19.4,  9828),
    (2017, 12.9, 18.9, 10088),
    (2018, 12.6, 18.4, 10287),
    (2019, 12.3, 18.0, 10890),
    (2020, 12.1, 17.7, 11312),
    (2021, 11.0, 16.0, 11908),
    (2022,  9.8, 14.3, 13545),
    (2023,  8.5, 12.4, 14339),
]
# train: 2010-2019 (10 obs), test: 2020-2023 (4 obs)
train4 = [r for r in poverty_all if r[0] <= 2019]
test4  = [r for r in poverty_all if r[0] >= 2020]

write_csv(f'{BASE}/dataset4_poverty_train.csv',
          train4, ['year','poverty_pct','poor_population_mln','poverty_threshold_rub'])
write_csv(f'{BASE}/dataset4_poverty_test.csv',
          test4,  ['year','poverty_pct','poor_population_mln','poverty_threshold_rub'])
print(f"DS4 saved: train={len(train4)}, test={len(test4)}")


# ============================================================
# DATASET 5 — LIFE EXPECTANCY AT BIRTH (years), annual 2010-2023
# Source: Rosstat / World Bank (worldbank.org) – confirmed by svspb.net
# svspb.net/danmark/zhizn.php?l=rossija
# ============================================================
life_exp_all = [
    # year, life_exp_total, life_exp_male, life_exp_female
    (2010, 68.8, 63.1, 74.8),
    (2011, 69.7, 64.0, 75.6),
    (2012, 70.1, 64.6, 75.8),
    (2013, 70.6, 65.1, 76.3),
    (2014, 70.7, 65.3, 76.5),
    (2015, 71.2, 65.9, 76.7),
    (2016, 71.7, 66.5, 77.1),
    (2017, 72.5, 67.5, 77.6),
    (2018, 72.7, 67.7, 77.8),
    (2019, 73.3, 68.2, 78.2),  # pre-COVID record
    (2020, 71.5, 66.5, 76.4),  # COVID shock
    (2021, 70.1, 65.5, 74.8),  # COVID peak mortality
    (2022, 72.8, 67.6, 77.8),  # post-COVID recovery
    (2023, 73.4, 68.4, 78.3),  # new record
]
# train: 2010-2019 (10 obs), test: 2020-2023 (4 obs)
train5 = [r for r in life_exp_all if r[0] <= 2019]
test5  = [r for r in life_exp_all if r[0] >= 2020]

write_csv(f'{BASE}/dataset5_lifeexp_train.csv',
          train5, ['year','life_exp_total','life_exp_male','life_exp_female'])
write_csv(f'{BASE}/dataset5_lifeexp_test.csv',
          test5,  ['year','life_exp_total','life_exp_male','life_exp_female'])
print(f"DS5 saved: train={len(train5)}, test={len(test5)}")


# ============================================================
# EXPERT REFERENCE LINKS (restricted to training period ≤ 2019)
# ============================================================
expert_refs = {
    "dataset1_unemployment": {
        "title": "Уровень безработицы в России (% по МОТ), 2010-2019",
        "target_variable": "unemployment_rate_pct",
        "train_period": "2010-2019",
        "test_period": "2020-2023",
        "links": [
            {
                "url": "http://ps.rosstat.gov.ru/unemployment",
                "description": "Росстат — Понятная статистика: Занятость и безработица (методология МОТ)",
                "coverage": "2010-2019"
            },
            {
                "url": "https://rosstat.gov.ru/labour_force",
                "description": "Росстат — Трудовые ресурсы, занятость и безработица (официальные данные)",
                "coverage": "2010-2019"
            },
            {
                "url": "https://svspb.net/danmark/bezrabotica.php?l=rossija",
                "description": "Сводная таблица уровня безработицы в России по годам",
                "coverage": "2010-2019"
            },
            {
                "url": "https://rosstat.gov.ru/storage/mediabank/Trud_2023.pdf",
                "description": "Росстат — Труд и занятость в России 2023 (PDF, исторические ряды)",
                "coverage": "2010-2019"
            }
        ]
    },
    "dataset2_natural_change": {
        "title": "Естественное движение населения РФ (тыс. чел.), 2010-2019",
        "target_variable": "nat_change_thous",
        "train_period": "2010-2019",
        "test_period": "2020-2023",
        "links": [
            {
                "url": "https://rosstat.gov.ru/compendium/document/13269",
                "description": "Росстат — Естественное движение населения (сборник абсолютных данных)",
                "coverage": "2010-2019"
            },
            {
                "url": "https://rosstat.gov.ru/folder/12781",
                "description": "Росстат — Демография (рождаемость, смертность, естественный прирост)",
                "coverage": "2010-2019"
            },
            {
                "url": "https://gogov.ru/articles/natural-increase",
                "description": "Gogov.ru — Смертность и рождаемость в России: статистика по годам",
                "coverage": "2010-2019"
            },
            {
                "url": "https://rosstat.gov.ru/storage/mediabank/Demogr_ejegod_2023.pdf",
                "description": "Демографический ежегодник России 2023 (исторические ряды с 2010 г.)",
                "coverage": "2010-2019"
            }
        ]
    },
    "dataset3_pensioners": {
        "title": "Численность пенсионеров РФ (млн чел.) и средняя пенсия (руб.), 2010-2019",
        "target_variable": "pensioners_mln",
        "train_period": "2010-2019",
        "test_period": "2020-2023",
        "links": [
            {
                "url": "https://rosstat.gov.ru/free_doc/new_site/population/urov/urov_p2.htm",
                "description": "Росстат — Численность пенсионеров и средний размер назначенных пенсий",
                "coverage": "2010-2019"
            },
            {
                "url": "https://sfr.gov.ru/info/statistics/pension_provision",
                "description": "СФР (ПФР) — Сведения о численности пенсионеров в системе ПФР",
                "coverage": "2010-2019"
            },
            {
                "url": "https://rosstat.gov.ru/folder/13877",
                "description": "Росстат — Старшее поколение: численность пенсионеров по видам пенсий",
                "coverage": "2010-2019"
            },
            {
                "url": "https://www.demoscope.ru/weekly/2023/0977/barom05.php",
                "description": "Demoscope Weekly — Численность пенсионеров 2010-2022 (аналитика)",
                "coverage": "2010-2019"
            }
        ]
    },
    "dataset4_poverty": {
        "title": "Уровень бедности в РФ (% населения ниже ПМ), 2010-2019",
        "target_variable": "poverty_pct",
        "train_period": "2010-2019",
        "test_period": "2020-2023",
        "links": [
            {
                "url": "https://rosstat.gov.ru/folder/13397",
                "description": "Росстат — Уровень жизни: реальные доходы и уровень бедности",
                "coverage": "2010-2019"
            },
            {
                "url": "https://gogov.ru/articles/standard-of-living",
                "description": "Gogov.ru — Уровень жизни в России: статистика бедности по годам",
                "coverage": "2010-2019"
            },
            {
                "url": "https://rosstat.gov.ru/storage/mediabank/Soc_pol_2023.pdf",
                "description": "Росстат — Социальное положение и уровень жизни населения России 2023",
                "coverage": "2010-2019"
            },
            {
                "url": "https://rosstat.gov.ru/free_doc/new_site/population/urov/urov_11g.htm",
                "description": "Росстат — Среднедушевые денежные доходы населения по годам",
                "coverage": "2010-2019"
            }
        ]
    },
    "dataset5_life_expectancy": {
        "title": "Ожидаемая продолжительность жизни при рождении (лет), Россия, 2010-2019",
        "target_variable": "life_exp_total",
        "train_period": "2010-2019",
        "test_period": "2020-2023",
        "links": [
            {
                "url": "https://rosstat.gov.ru/folder/12781",
                "description": "Росстат — Демография: ожидаемая продолжительность жизни при рождении",
                "coverage": "2010-2019"
            },
            {
                "url": "https://svspb.net/danmark/zhizn.php?l=rossija",
                "description": "Сводная таблица ОПЖ в России по годам (Мировой банк / Росстат)",
                "coverage": "2010-2019"
            },
            {
                "url": "https://rosstat.gov.ru/bgd/regl/b07_13/isswww.exe/stg/d01/04-23.htm",
                "description": "Росстат — Таблица 4.23 ОПЖ при рождении: исторический ряд",
                "coverage": "2010-2019"
            },
            {
                "url": "https://eng.rosstat.gov.ru/storage/mediabank/DEM23.pdf",
                "description": "Демографический ежегодник России 2023 (EN/RU, исторические ряды)",
                "coverage": "2010-2019"
            }
        ]
    }
}

with open(f'{BASE}/expert_references.json', 'w', encoding='utf-8') as f:
    json.dump(expert_refs, f, ensure_ascii=False, indent=2)

print("\nAll datasets and expert_references.json created successfully.")
print("\nSummary:")
for ds_key, ds_val in expert_refs.items():
    print(f"  {ds_key}: {ds_val['title']}")

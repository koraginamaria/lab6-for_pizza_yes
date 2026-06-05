# lab6-for_pizza_yes

Репозитории 6 лабораторной работы команды: за пиццу ДА!

Тут лежат данные https://disk.yandex.ru/d/LUQAjPM_vbnCfg

В mobilnet9189_1.ipynb код обучения модели 

# Инструкция по запуску

1. Скачайте  файлы весов фолдов (`model_fold_1.pth` ... `model_fold_5.pth`) с релиза гита.
   
2. Поместите скачанные файлы весов в папку `weights/` внутри репозитория.

3. В корне проекта выполните команду сборки:
   ```bash
   docker build -t employee_action_classifier .

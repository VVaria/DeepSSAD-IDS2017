# DeepSSAD-IDS2017

Данные для обучения и тестирования: https://drive.google.com/drive/folders/1m35EIh_ApY0acA2DjDFFQxBkh85pOwQV?usp=sharing

Для запуска проекта Вам потребуется скачать данные из Google Disk
Папку Model положить в корень, csv-файлы перенести в папку Data.

Результаты Вы можете увидеть в папке exps.



Метрика AUC ROC для каждого класса и типа модели
| Класс данных\Тип модели |  AE  | AE-SAD | VAE  | VAE-SAD |
| ----------------------- |:----:|:------:|:----:|:-------:|
| Bot                     | 0,14 |  0,79  | 0,69 | 0,79    |
| DDoS                    | 0,86 |  0,91  | 0,91 | 0,97    | 
| DoS GoldenEye           | 0,97 |  0,99  | 0,93 | 0,97    |
| DoS Hulk                | 0,79 |  1,00  | 0,91 | 0,99    |
| DoS Slowhttptest        | 0,97 |  0,97  | 0,96 | 0,98    |
| DoS slowloris           | 0,64 |  0,98  | 0,88 | 0,95    |
| FTP-Patator             | 0,51 |  0,98  | 0,79 | 0,87    |
| Heartbleed              | 1,00 |  0,95  | 1,00 | 1,00    |
| Infiltration            | 0,93 |  0,40  | 0,94 | 0,70    |
| SSH-Patator             | 0,49 |  0,91  | 0,68 | 0,82    |
| Web Attack Brute Force  | 0,37 |  0,23  | 0,80 | 0,83    |
| Web Attack Sql Injection| 0,54 |  0,81  | 0,67 | 0,85    |
| Web Attack XSS          | 0,36 |  0,21  | 0,82 | 0,83    |
| **Все классы**              | **0,78** |  **0,97**  | **0,90** | **0,98**    |


-------------

Training and Test Data: https://drive.google.com/drive/folders/1m35EIh_ApY0acA2DjDFFQxBkh85pOwQV?usp=sharing

To start the project, you need to download data from Google Disk.
Put the Model folder in the root, move the csv files to the Data folder.

You can see the results in the exps folder. 



AUC ROC metric for each class and type of model
|  Data class\Model Type  |  AE  | AE-SAD | VAE  | VAE-SAD |
| ----------------------- |:----:|:------:|:----:|:-------:|
| Bot                     | 0,14 |  0,79  | 0,69 | 0,79    |
| DDoS                    | 0,86 |  0,91  | 0,91 | 0,97    | 
| DoS GoldenEye           | 0,97 |  0,99  | 0,93 | 0,97    |
| DoS Hulk                | 0,79 |  1,00  | 0,91 | 0,99    |
| DoS Slowhttptest        | 0,97 |  0,97  | 0,96 | 0,98    |
| DoS slowloris           | 0,64 |  0,98  | 0,88 | 0,95    |
| FTP-Patator             | 0,51 |  0,98  | 0,79 | 0,87    |
| Heartbleed              | 1,00 |  0,95  | 1,00 | 1,00    |
| Infiltration            | 0,93 |  0,40  | 0,94 | 0,70    |
| SSH-Patator             | 0,49 |  0,91  | 0,68 | 0,82    |
| Web Attack Brute Force  | 0,37 |  0,23  | 0,80 | 0,83    |
| Web Attack Sql Injection| 0,54 |  0,81  | 0,67 | 0,85    |
| Web Attack XSS          | 0,36 |  0,21  | 0,82 | 0,83    |
| **All classes**              | **0,78** |  **0,97**  | **0,90** | **0,98**    |



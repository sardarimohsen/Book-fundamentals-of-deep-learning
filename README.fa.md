<div align="center">

**🌐 [English](./README.md) | [فارسی](./README.fa.md)**

</div>

<div align="center">

# 📘 مبانی یادگیری عمیق
### Fundamentals of Deep Learning

*مخزن همراه کتاب — کدهای منبع و تصاویر مربوط به فصل‌های کتاب*

<br>

[![Python](https://img.shields.io/badge/Python-3.12-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![Keras](https://img.shields.io/badge/Keras-D00000?style=for-the-badge&logo=keras&logoColor=white)](https://keras.io/)

[![Stars](https://img.shields.io/github/stars/sardarimohsen/Book-fundamentals-of-deep-learning?style=flat-square&color=yellow)](../../stargazers)
[![Forks](https://img.shields.io/github/forks/sardarimohsen/Book-fundamentals-of-deep-learning?style=flat-square&color=blue)](../../network/members)
[![Issues](https://img.shields.io/github/issues/sardarimohsen/Book-fundamentals-of-deep-learning?style=flat-square&color=orange)](../../issues)
[![Last Commit](https://img.shields.io/github/last-commit/sardarimohsen/Book-fundamentals-of-deep-learning?style=flat-square&color=brightgreen)](../../commits)

</div>

<br>

> ⚠️ **این مخزن شامل متن کامل کتاب نیست.**
> اینجا فقط **کدهای منبع و تصاویر** مربوط به کتاب قرار داره. برای دریافت نسخه‌ی کامل (PDF) یا خرید نسخه‌ی چاپی، به بخش [📥 دریافت کتاب](#-دریافت-کتاب) در پایین همین صفحه مراجعه کن.

<br>

## 📑 فهرست مطالب

- [درباره کتاب](#-درباره-کتاب)
- [فصل‌ها](#-فصل‌ها)
- [ساختار ریپازیتوری](#-ساختار-ریپازیتوری)
- [پیش‌نیازها](#️-پیش‌نیازها)
- [شروع کار](#-شروع-کار)
- [غلط‌نامه](#-غلط‌نامه)
- [دریافت کتاب](#-دریافت-کتاب)
- [ارجاع (Citation)](#-ارجاع-citation)
- [نویسندگان](#️-نویسندگان)

<br>

## 📖 درباره کتاب

کتاب **مبانی یادگیری عمیق** خواننده رو از مبانی ریاضی و یادگیری ماشین شروع می‌کنه و تا معماری‌های مدرن یادگیری عمیق پیش می‌بره — با تأکید زیاد بر **یادگیری از طریق تمرین عملی**. هر فصل عملی کتاب با کد اجرایی متناظر در همین ریپازیتوری همراهه تا بتونی همزمان با مطالعه، کد رو اجرا و تجربه کنی.

<br>

## 📚 فصل‌ها

<table>
<tr><th>#</th><th>عنوان فصل</th><th>کد</th></tr>
<tr><td>۱</td><td>مقدمه‌ای بر هوش مصنوعی و یادگیری عمیق</td><td align="center">—</td></tr>
<tr><td>۲</td><td>مفاهیم پایه یادگیری ماشین</td><td align="center">—</td></tr>
<tr><td>۳</td><td>ریاضیات موردنیاز یادگیری عمیق</td><td align="center">—</td></tr>
<tr><td>۴</td><td>شبکه‌های عصبی مصنوعی</td><td align="center">—</td></tr>
<tr><td>۵</td><td>آموزش شبکه‌های عصبی</td><td align="center">—</td></tr>
<tr><td>۶</td><td>توابع فعال‌سازی</td><td align="center">—</td></tr>
<tr><td>۷</td><td>آشنایی با TensorFlow و Keras</td><td align="center">✅</td></tr>
<tr><td>۸</td><td>ارزیابی مدل‌های یادگیری عمیق</td><td align="center">✅</td></tr>
<tr><td>۹</td><td>پردازش داده‌ها</td><td align="center">✅</td></tr>
<tr><td>۱۰</td><td>شبکه‌های عصبی کانولوشنی (CNN)</td><td align="center">✅</td></tr>
<tr><td>۱۱</td><td>شبکه‌های عصبی بازگشتی (RNN)</td><td align="center">✅</td></tr>
</table>

<sub>فصل‌های ۱ تا ۶ پایه‌های نظری و ریاضی رو پوشش می‌دن؛ فصل‌های ۷ تا ۱۱ همراه با کد کامل و اجرایی هستن.</sub>

<br>

## 🗂 ساختار ریپازیتوری

```
Book-fundamentals-of-deep-learning/
│
├── code/                              # کدهای منبع (.py)، به‌تفکیک هر فصل
│   ├── chapter07_tensorflow_keras/
│   ├── chapter08_model_evaluation/
│   ├── chapter09_data_processing/
│   ├── chapter10_cnn/
│   └── chapter11_rnn/
│
├── images/                            # تصاویر و نمودارهای کتاب
│
├── errata/                            # اصلاحات و غلط‌نامه
│
└── README.md
```

<br>

## ⚙️ پیش‌نیازها

| کتابخانه |
|---|
| Python 3.12 |
| TensorFlow |
| NumPy |
| Matplotlib |
| Scikit-learn |

```bash
pip install -r code/requirements.txt
```

<br>

## 🚀 شروع کار

```bash
# ۱. کلون کردن ریپازیتوری
git clone https://github.com/sardarimohsen/Book-fundamentals-of-deep-learning.git
cd Book-fundamentals-of-deep-learning

# ۲. نصب پیش‌نیازها
pip install -r code/requirements.txt

# ۳. اجرای کد یک فصل
python code/chapter07_tensorflow_keras/example.py
```

<br>

## 🐞 غلط‌نامه

اگه در کتاب به اشتباهی برخوردی، اول پوشه‌ی [`errata/`](./errata) رو چک کن. اگه اونجا ثبت نشده بود، لطفاً یک [issue جدید](../../issues) باز کن — این کار به بهبود چاپ‌های بعدی کتاب کمک می‌کنه.

<br>

## 📥 دریافت کتاب

این ریپازیتوری فقط **بخش کد** کتاب رو پوشش می‌ده. برای دریافت **متن کامل کتاب**، یکی از گزینه‌های زیر رو انتخاب کن:

<div align="center">

| می‌خوام... | تماس |
|---|---|
| 📄 فایل **PDF** کتاب رو دریافت کنم | [![Telegram](https://img.shields.io/badge/Telegram-@Comp__sardari-26A5E4?style=for-the-badge&logo=telegram&logoColor=white)](https://t.me/Comp_sardari) [![Telegram](https://img.shields.io/badge/Telegram-@S__M__A__KH-26A5E4?style=for-the-badge&logo=telegram&logoColor=white)](https://t.me/S_M_A_KH) |
| 📦 نسخه‌ی **چاپی** کتاب رو تهیه کنم | [![Telegram](https://img.shields.io/badge/Telegram-@Comp__sardari-26A5E4?style=for-the-badge&logo=telegram&logoColor=white)](https://t.me/Comp_sardari) [![Telegram](https://img.shields.io/badge/Telegram-@S__M__A__KH-26A5E4?style=for-the-badge&logo=telegram&logoColor=white)](https://t.me/S_M_A_KH) |

</div>

کافیه به یکی از آیدی‌های بالا پیام بدی تا نویسندگان راهنماییت کنن.

<br>

## 📄 ارجاع (Citation)

> به‌زودی.

<br>

## ✍️ نویسندگان

<table>
<tr>
<td align="center">
<b>محسن سرداری زرچی</b><br>
<a href="https://t.me/Comp_sardari">
<img src="https://img.shields.io/badge/Telegram-26A5E4?style=flat-square&logo=telegram&logoColor=white" />
</a>
</td>
<td align="center">
<b>سید محمد آرمان خلیلی</b><br>
<a href="https://t.me/S_M_A_KH">
<img src="https://img.shields.io/badge/Telegram-26A5E4?style=flat-square&logo=telegram&logoColor=white" />
</a>
</td>
</tr>
</table>

<br>

<div align="center">

**⭐ اگه این ریپازیتوری بهت کمک کرد، یک ستاره بهش بده!**

</div>

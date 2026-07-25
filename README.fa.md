<div align="center">

**🌐 [English](./README.md) | [فارسی](./README.fa.md)**

</div>

<div align="center">

# 📘 مبانی یادگیری عمیق

*ریپازیتوری همراه کتاب — کدهای منبع و تصاویر مربوط به فصل‌های کتاب*

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

> ⚠️ **این ریپازیتوری شامل متن کامل کتاب نیست.**
> اینجا فقط **کدهای منبع و تصاویر** مربوط به کتاب قرار دارد. برای دریافت نسخه‌ی کامل (PDF) یا خرید نسخه‌ی چاپی، به بخش [📥 دریافت کتاب](#-دریافت-کتاب) در پایین همین صفحه مراجعه کنید.

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

کتاب **مبانی یادگیری عمیق** خواننده را از مبانی ریاضی و یادگیری ماشین تا معماری‌های مدرن یادگیری عمیق همراهی می‌کند؛ با تأکید زیاد بر **یادگیری از طریق تمرین عملی**. هر فصل عملی کتاب با کد اجرایی متناظر در همین ریپازیتوری همراه است تا بتوانید همزمان با مطالعه، کد رو اجرا و تجربه کنید.

<br>

## 📚 فصل‌ها
<div align="right">

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

</div>

<sub>فصل‌های ۱ تا ۶ پایه‌های نظری و ریاضی رو پوشش می‌دهند؛ فصل‌های ۷ تا ۱۱ همراه با کد کامل و اجرایی هستند.</sub>

<br>

## 🗂 ساختار ریپازیتوری

```
Book-fundamentals-of-deep-learning/
│
├── code/                              # Source code organized by chapter
│   ├── Chapter 7 - Introduction to TensorFlow and Keras/
│   ├── Chapter 8 - Model Evaluation Metrics/
│   ├── Chapter 9 - Data Preprocessing/
│   ├── Chapter 10 - Convolutional Neural Networks/
│   └── Chapter 11 - Recurrent Neural Networks/
│
├── images/                            # Figures & diagrams used in the book
│
├── errata/                            # Corrections and clarifications
│
└── README.md
```

<br>

## ⚙️ پیش‌نیازها
<div align="right">

| کتابخانه |
|---|
| Python 3.12 |
| TensorFlow |
| NumPy |
| Matplotlib |
| Scikit-learn |

</div>

```bash
pip install -r code/requirements.txt
```

<br>

## 🚀 شروع کار

```bash
# 1. Clone the repository
git clone https://github.com/sardarimohsen/Book-fundamentals-of-deep-learning.git
cd Book-fundamentals-of-deep-learning

# 2. Install dependencies
pip install -r code/requirements.txt

# 3. Launch a chapter notebook
python code/Chapter 7 - Introduction to TensorFlow and Keras/Chapter_07_tensorflow_keras.py
```

<br>

## 🐞 غلط‌نامه

اگر در کتاب به اشتباهی برخوردید، اول پوشه‌ی [`errata/`](./errata) را چک کنید. اگر آنجا ثبت نشده بود، لطفاً یک [issue جدید](../../issues) باز کنید. این کار به بهبود چاپ‌های بعدی کتاب کمک می‌کند.

<br>

## 📥 دریافت کتاب

این ریپازیتوری فقط **بخش کد** کتاب را پوشش می‌دهد. برای دریافت نسخه PDF یا نسخه فیزیکی **کتاب**، به یکی از آیدی‌های زیر در تلگرام پیام بدهید:

<div align="center">
  
[![Telegram](https://img.shields.io/badge/Telegram-@Comp__sardari-26A5E4?style=for-the-badge&logo=telegram&logoColor=white)](https://t.me/Comp_sardari) 
[![Telegram](https://img.shields.io/badge/Telegram-@S__M__A__KH-26A5E4?style=for-the-badge&logo=telegram&logoColor=white)](https://t.me/S_M_A_KH)

</div>


<br>

## 📄 ارجاع (Citation)

> به‌زودی.

<br>

## ✍️ نویسندگان

* محسن سرداری زارچی
* سید محمد آرمان خلیلی

<br>

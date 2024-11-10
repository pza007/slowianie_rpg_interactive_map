# Copyright (c) 2024, Przemyslaw Zawadzki
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from enum import Enum
import datetime
from dateutil.relativedelta import relativedelta


class EnumTimeOfDay(Enum):
    morning = 1
    noon    = 2
    evening = 3
    night   = 4


def get_event_title(val):
    if val == 1:    return ""
    elif val == 2:  return "Chram"
    elif val == 3:  return "Święte miejsce"
    elif val == 4:  return "Obóz robotników"
    elif val == 5:  return "Domostwo farmera"
    elif val == 6:  return "Postrzyżyny"
    elif val == 7:  return "Swaćba - obrzęd"
    elif val == 8:  return "Swaćba - druhowie"
    elif val == 9:  return "Pochówek"
    elif val == 10: return "Spotkanie Krwawicy"
    elif val == 11: return "Wędrująca Bieda"
    elif val == 12: return "Chodzące Licho"
    elif val == 13: return "Chłop"
    elif val == 14: return "Drobny kupiec"
    elif val == 15: return "Włóczęga"
    elif val == 16: return "Biedny bajarz"
    elif val == 17: return "Ślepiec"
    elif val == 18: return "Wędrowny głupek"
    elif val == 19: return "Pijak"
    elif val == 20: return "Hazardzista"
    elif val == 21: return "Siłacz"
    elif val == 22: return "Nożownik"
    elif val == 23: return "Złodziejaszek"
    elif val == 24: return "Szczęśliwy kamyk"
    elif val == 25: return "Dobre duszki"
    elif val == 26: return "Samotne drzewo"
    elif val == 27: return "Martwe zwierzę"
    elif val == 28: return "Złe myśli"
    elif val == 29: return "Zły omen"
    elif val == 30: return "Opuszczona chatka"
    elif val == 31: return "Źródełko"
    elif val == 32: return "Urodzajna polana"
    elif val == 33: return "Mała grota"
    elif val == 34: return "Mały zajazd"
    elif val == 35: return "Kręgi kamienne"
    elif val == 36: return "Samotne domostwo"
    elif val == 37: return "Ognisko"
    elif val == 38: return "Leże stworka"
    elif val == 39: return "Obóz nieznajomych"
    elif val == 40: return "Miejsce rytualne"
    elif val == 41: return "Oprychy"
    elif val == 42: return "Bałamutnik"
    elif val == 43: return "Cholerniki"
    elif val == 44: return "Błotnik"
    elif val == 45: return "Kocmołuch"
    elif val == 46: return "Umrzyki"
    elif val == 47: return "Wężyr Wielki"
    elif val == 48: return "Niedobitki Germanów"
    elif val == 49: return "Ludzie Lasu"
    elif val == 50: return "Dzikusy"
    elif val == 51: return "Inne drużyny"
    elif val == 52: return "Zarwany most"
    elif val == 53: return "Atak choroby"
    elif val == 54: return "Zgubione przedmioty"
    elif val == 55: return "Pomylone ścieżki"
    elif val == 56: return "Możliwy skrót"
    elif val == 57: return "Pognite jadło"
    elif val == 58: return "Skowyt duchów"
    elif val == 59: return "Dzika zwierzyna"
    elif val == 60: return "Zajazd strzeżony"
    elif val == 61: return "Wężowij"
    elif val == 62: return "Gryf"
    elif val == 63: return "Kaduk"
    elif val == 64: return "Kuk"
    elif val == 65: return "Leszy"
    elif val == 66: return "Błędnica"
    elif val == 67: return "Hermus"
    elif val == 68: return "Harcuk"
    elif val == 69: return "Upolowane zwierzę"
    elif val == 70: return "Zwłoki"
    elif val == 71: return "Wisielec"
    elif val == 72: return "Wnyki"
    elif val == 73: return "Powalone drzewo"
    elif val == 74: return "Ostrokrzew trujący"
    elif val == 75: return "Miejsce mordu"
    elif val == 76: return "Słaby podest"
    elif val == 77: return "Posłuch"
    elif val == 78: return "Niech wam darzą"
    elif val == 79: return "Dobre duchy z wami"
    elif val == 80: return "Przychylność"
    elif val == 81: return "Posłaniec bogów"
    elif val == 82: return "Nauczyciel"
    elif val == 83: return "Zyskanie reputacji"
    elif val == 84: return "Utrata reputacji"
    elif val == 85: return "Komes"
    elif val == 86: return "Bajarz"
    elif val == 87: return "Kupiec"
    elif val == 88: return "Myśliwy"
    elif val == 89: return "Łowca głów"
    elif val == 90: return "Najemnik"
    elif val == 91: return "Wołchw"
    elif val == 92: return "Wartownik"
    elif val == 93: return "Zwierzęcy strażnik"
    elif val == 94: return "Zabójca potworów"
    elif val == 95: return "Złodziej"
    elif val == 96: return "Żerca"
    elif val == 97: return "Dobry omen"
    elif val == 98: return "Brzozy lecznicze"
    elif val == 99: return "Amulet"
    else:           return "Pomyślne wiatry"


_date = datetime.date(2023, 12, 22)
y_pos = {(22,12): 0}
for idx in range(1, 366):
    _date = _date + relativedelta(days=1)
    d, m = _date.day, _date.month
    y_pos[(d, m)] = round(idx*1.31147)

def get_time_pointer_position(day, month):
    return y_pos[day, month]
from pofi.text import normalize_for_tts, number_to_words, ordinal_words, tr_lower, tr_upper


def test_turkish_case():
    assert tr_upper("iğne ılık") == "İĞNE ILIK"
    assert tr_lower("IŞIK İzmir") == "ışık izmir"


def test_numbers():
    assert number_to_words(0) == "sıfır"
    assert number_to_words(1923) == "bin dokuz yüz yirmi üç"
    assert number_to_words(100) == "yüz"
    assert number_to_words(2026) == "iki bin yirmi altı"
    assert number_to_words(1000000) == "bir milyon"


def test_ordinals():
    assert ordinal_words(1) == "birinci"
    assert ordinal_words(3) == "üçüncü"
    assert ordinal_words(6) == "altıncı"
    assert ordinal_words(10) == "onuncu"


def test_normalize():
    assert normalize_for_tts("2 + 3 = 5") == "iki artı üç eşittir beş"
    assert normalize_for_tts("1923'te") == "bin dokuz yüz yirmi üçte"
    assert normalize_for_tts("Merhaba 🎉 çocuklar!") == "Merhaba çocuklar!"
    assert normalize_for_tts("%50 indirim") == "yüzde elli indirim"

from scripts.evaluate_asr import character_errors, normalized_text


def test_character_error_distance_preserves_meaningful_differences():
    assert character_errors("不要开灯", "要开灯") == 1
    assert character_errors("小明", "小名") == 1
    assert character_errors("123", "124") == 1
    assert character_errors("", "abc") == 3
    assert character_errors("abc", "") == 3
    assert normalized_text("ＡＢＣ，123！") == "abc123"

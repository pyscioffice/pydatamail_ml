import bleach
import cssutils
from langdetect import DetectorFactory, detect, LangDetectException
import logging
import re

# Fixes
DetectorFactory.seed = 0
cssutils.log.setLevel(logging.CRITICAL)


def detect_language(df):
    if df["clean"] is not None:
        try:
            return detect(df["clean"])
        except LangDetectException:
            return None
    else:
        return None


def get_bleach_text(text_input):
    return bleach.clean(text_input)


def clean_str(text_input):
    for special_character in ["\n", "&lt;", "&gt;", "&amp;", "_"]:
        text_input = text_input.replace(special_character, " ")
    return text_input


def text_remove_whitespace(text_input):
    return " ".join(text_input.split())


def get_text_after_last_css_tag(text_input):
    css_info = cssutils.parseString(text_input)
    css_str = css_info.cssText.decode()
    css_str_clean = clean_str(text_input=css_str)
    css_str_lst = css_str_clean.split(":")
    if len(css_str_lst) > 3:
        css_lst_tag = css_str_clean.split(":")[-2].split()[-1]
        return text_input.split(css_lst_tag)[-1]
    else:
        return text_input


def text_remove_number(text_input):
    return "".join([i for i in text_input if not i.isdigit()])


def text_remove_url(text_input):
    return re.sub(r"http\S+", "", text_input)


def text_pipeline(text_input):
    if text_input is not None:
        text_bleach = get_bleach_text(text_input=text_input)
        text_clean = clean_str(text_input=text_bleach)
        text_nocss = get_text_after_last_css_tag(text_input=text_clean)
        text_nourl = text_remove_url(text_input=text_nocss)
        text_single_whitespace = text_remove_whitespace(text_input=text_nourl)
        return text_remove_number(text_input=text_single_whitespace)
    else:
        return text_input

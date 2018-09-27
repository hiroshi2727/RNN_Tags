'''This file is used to make a csv file for personal dictionary of mecab'''
import MeCab

tagger = MeCab.Tagger("-Owakati")
tags_for_dict = []
i = 0
for tag in list_tags:
    if tag != "":
        parsed_tag = tagger.parse(tag).split()
        try:
            if len(parsed_tag) != 1:
                tags_for_dict.append(tag)
        except KeyError:
            pass

for tag in tags_for_dict[:10]:
    print(tag+",,,0,名詞,一般,*,*,*,*,*,*,*")

file_to_save.to_csv("for_user_dict.csv")
tags_to_save = pd.Series(tags_for_dict).T
tags_to_save.to_csv("./for_user_dic.csv")

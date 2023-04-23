#--------------------------------------------------------------------
# Shows how to strip brackets using regular expressions in Python
# Based on https://stackoverflow.com/questions/42324466/python-regular-expression-to-remove-all-square-brackets-and-their-contents
#--------------------------------------------------------------------
import re

sentence = "Officially the Swiss Confederation, it is a country situated in the confluence of western, central, and southern Europe.[9][note 4] It is a federal republic composed of 26 cantons"
print("\nHere is the sentence with brackets")
print(sentence)

regex_no_brackets = r'\[.*?\]'
stripped_sentence = re.sub(regex_no_brackets, '', sentence)
print("\nHere is the sentence with brackets stripped out")
print(stripped_sentence)



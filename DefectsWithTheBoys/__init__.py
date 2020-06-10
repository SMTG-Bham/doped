from datetime import date

# print("""I'm unapologetic you can tell by what I'm on.
# If you don't like this, fuck you, this is nothing but a song.\n""")
if date.today().weekday() in [5, 6]:
    print("""Working on the weekend, like usual...\n""")
if date.today().weekday() == 5:
    print("Seriously though, everyone knows Saturday's for the boys!\n")

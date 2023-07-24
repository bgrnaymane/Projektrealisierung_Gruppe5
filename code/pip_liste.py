import subprocess

try:
    result = subprocess.check_output(["pip", "list"])
    installed_packages = result.decode("utf-8")
except subprocess.CalledProcessError as e:
    print("Fehler beim Abrufen der installierten Pakete:", e)
    exit(1)

# Die Liste der installierten Pakete in eine Textdatei speichern
with open("installed_packages.txt", "w") as file:
    file.write(installed_packages)

print("Die Liste der installierten Pakete wurde erfolgreich in 'installed_packages.txt' gespeichert.")

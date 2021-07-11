# AB Restauration PWA

AB Restauration is a company that specifies on logistic of large events in France.
This project aim to create a PWA aimed towards logistic and distribution
As of yet, the project focus only on the Stade de la Meinau Strasbourg France. but is concieved to scale easily to other stadium.


# How to use ?

The whole Django is kept inside the PWA folder.
All information can be found there.
The folder contains a Readme dedicated to the original github project.

## To run the script, you must :

prepare a virtual environnement, run it. then be sure to install requirements :

```pip install -r requirements.txt```

select the PWA folder in terminal (depends on where you store it locally),
run the virtual environnement then run the command :

```python manage.py runserver```

or depending on python version

```python3 manage.py runserver```


sometimes you need to check migrations

```python manage.py migrate --run-syncdb```

```python manage.py makemigration```


# User Registration

The project includes a user registration and login system.
This will determine the limit or lack there of acces to the app
already working and usable.

### TODO : need to send the email for confirmation

# Meteo

meteo.py is a functional module that allows to return the weather at a precise date and time


# Usefull links :

A Mural with the product backlog available:
[Mural Link](https://app.mural.co/t/userstorywallofdeath3647/m/userstorywallofdeath3647/1620113036127/c20ec6dd86749b0d9f96669d1dea4975f93782a4)

A Trello is made to follow the sprints:
[Trello Link](https://trello.com/b/oBE7Cx0Z/ab-burndown-chart)
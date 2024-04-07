curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
pip install -r requirements.txt
python3.9 manage.py collectstatic --noinput
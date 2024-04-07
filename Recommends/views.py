import time
import re
import numpy as np 
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from keras.models import load_model
from datetime import datetime
import nltk
from nltk.corpus import stopwords
from wordcloud import WordCloud, STOPWORDS
from nltk.stem.wordnet import WordNetLemmatizer

import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

import warnings
warnings.filterwarnings('ignore')

import csv
import langid
import requests
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.service import Service as ChromeService
from selenium.webdriver.chrome.options import Options

from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.cache import never_cache
from django.http import HttpResponse, JsonResponse
from django.conf import settings

# model = load_model('D:/senti_pro/ECommerce/Recommends/static/SavedFiles/bilstm_model.h5')
model = load_model('Recommends/bilstm_model.h5')
# D:\senti_pro\github\django-sentiscope\Recommends\bilstm_model.h5

# D:/senti_pro/github/django-sentiscope/bilstm_model.h5
product1 = predict1 = range1 = popular1 = price1 = site1 = ''
product2 = predict2 = range2 = popular2 = price2 = site2 = ''

# Create your views here.
def login_page(request):
    return render(request,'login.html')

def login(request):
    username = request.POST.get('username')
    password = request.POST.get('password')

    if username == 'admin' and password == 'admin':
        request.session['user'] = username
        name = username.capitalize()
        return render(request,'home.html',{'user':name})
    else:
        return HttpResponse("<script>alert('Login Failed!!...');window.location.href='/login_page/'</script>")
    
@never_cache
def home_page(request):
    if 'user' in request.session:
        name = request.session['user'].capitalize()
        return render(request,'home.html',{'user':name})


@never_cache     
def logout(request):        # logout
    if 'user' in request.session:
        del request.session['user']
        # del request.session['request_id']
    return render(request,'login.html')

def analyse_page(request):
    return render(request,'analyse.html')


# Function to check if a text is in English
def is_english(text):
    lang, _ = langid.classify(text)
    return lang == 'en'

def generate_amazon_url(product_id):
    # base_url = 'https://www.amazon.in/s?k='
    # base_url = f'https://www.amazon.in/{product_name}/dp/{product_id}/?th=1'
    base_url = f'https://www.amazon.co.in/product-reviews/{product_id}'
    return f'{base_url}'




def scrape_and_save_reviews(product_url,):
    HEADERS = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36', 'Accept-Language': 'en-US, en;q=0.5'}
    webpage = requests.get(product_url, headers=HEADERS)
    if webpage.status_code == 200:
      print('Response: ',webpage.status_code) 
      soup = BeautifulSoup(webpage.content, 'html.parser')
      return soup
    else:
      print('Response: ',webpage.status_code)
      return webpage
    

def scrape_and_get_price(product_url):
    HEADERS = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36', 'Accept-Language': 'en-US, en;q=0.5'}
    webpage = requests.get(product_url, headers=HEADERS)
    # webpage = requests.get(product_url)

    if webpage.status_code == 200:
      soup = BeautifulSoup(webpage.content, 'html.parser')
      return soup
    else:
      return webpage


def generate_c2_url(product_name):
    base_url = f'https://www.flipkart.com/search?q={product_name}'

    encoded_product_name = product_name.replace(' ', '%20')
    return f'{base_url} '

def generate_c3_url(product_name):
    base_url = f'http://shopping.indiamart.com/search.php?ss={product_name}'
    encoded_product_name = product_name.replace(' ', '%20')
    return f'{base_url} '


def input(product_name,pid,min_price,max_price):
    print('inside input(product_name,pid,min,max)')

    start = time.time()
    

    product = product_name 
    product_id = pid
    
    max_ = max_price 
    min_ = min_price 
    amazon_url = generate_amazon_url(product_id)
    print(amazon_url)
    

    amazon_url2 = f'https://www.amazon.in/product-reviews/{product_id}/ref=cm_cr_arp_d_paging_btm_next_2?pageNumber=2'


    chrome_options = Options()
    chrome_options.add_argument("--headless") 

    # Specify the path to your ChromeDriver recently downoloaded
    # chrome_path = 'D:/senti_pro/ECommerce/Recommends/static/chromedriver/chromedriver.exe' # Specify the path to your ChromeDriver recently downoloaded
    chrome_path = 'Recommends/static/chromedriver/chromedriver.exe' # Specify the path to your ChromeDriver recently downoloaded
    
    service = ChromeService(chrome_path)

   
    driver = webdriver.Chrome(service=service, options=chrome_options)

    try:
        
        driver.get(amazon_url)

        
        for i in range(5):
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(2)  

       
        page_source = driver.page_source
        soup = BeautifulSoup(page_source, 'html.parser')
        # print(soup)
        print('-'*50)

       
        reviews = soup.find_all('span', attrs={'data-hook': 'review-body'})
        profile_names = soup.find_all('div', class_='a-profile-content')
        rating = soup.find_all('span', class_='a-icon-alt')
        verfication = soup.find_all('span', class_='a-size-mini a-color-state a-text-bold')
        dates = soup.find_all('span', class_='a-size-base a-color-secondary review-date')
        print(dates)
        print(rating)
       
        review_list = []
        users_list= []
        rating_list = []
        varification_list = []
        date_list = []
        for review in reviews:
       
            inner_elements = review.find_all('span')

            for inner_element in inner_elements:
                review_list.append(inner_element.get_text(strip=True))
        filtered_reviews = [review for review in review_list if review]
        filtered_reviews = [review for review in filtered_reviews if is_english(review)]
        # print('First page reviews: \n' , filtered_reviews)

        for name in profile_names:
    
            user_names = name.find_all('span')

            for user_name in user_names:
                users_list.append(user_name.get_text(strip=True))
        print("names",users_list)
        for star in rating:
            match = re.search(r'(\d+\.\d+)', str(star))
            if match:
                rating = float(match.group())
            rating_list.append(rating)
        print("rating",rating_list)
        for verifi in verfication:
                varification_list.append(verifi.get_text(strip=True))
        print("varifi",varification_list)
        for date in dates:
            print(date)
            date = str(date)
            pattern = r'Reviewed in India on (\d{1,2} \w+ \d{4})'

            match = re.search(pattern, date)

            review_date = match.group(1)
            review_date = datetime.strptime(review_date, "%d %B %Y").date()
            print(review_date)
            date_list.append(review_date)


        if reviews:

            driver.get(amazon_url2)


            for i in range(5):
                driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
                time.sleep(2)

            page_source2 = driver.page_source
            soup2 = BeautifulSoup(page_source2, 'html.parser')
            # print(soup2)

            print('-'*50)

            reviews2 = soup2.find_all('span', attrs={'data-hook': 'review-body'})
            profile_names2 = soup2.find_all('div', class_='a-profile-content')
            rating = soup2.find_all('span', class_='a-icon-alt')
            verfication = soup2.find_all('span', class_='a-size-mini a-color-state a-text-bold')
            dates = soup2.find_all('span', class_='a-size-base a-color-secondary review-date')
            for review in reviews2:
                inner_elements = review.find_all('span')

                for inner_element in inner_elements:
                    review_list.append(inner_element.get_text(strip=True))
            filtered_reviews = [review for review in review_list if review]
            filtered_reviews = [review for review in filtered_reviews if is_english(review)]
            # print('Second page reviews: \n' , filtered_reviews)

            for name in profile_names2:
        
                user_names = name.find_all('span')

                for user_name in user_names:
                    users_list.append(user_name.get_text(strip=True))
            # print("names",users_list)
            for star in rating:
                match = re.search(r'(\d+\.\d+)', str(star))
                if match:
                    rating = float(match.group())
                rating_list.append(rating)
            print("rating",rating_list)
            for verifi in verfication:
                    varification_list.append(verifi.get_text(strip=True))
            print("varifi",varification_list)
            for date in dates:
                print(date)
                date = str(date)
                pattern = r'Reviewed in India on (\d{1,2} \w+ \d{4})'

                match = re.search(pattern, date)

                review_date = match.group(1)
                review_date = datetime.strptime(review_date, "%d %B %Y").date()
                print(review_date)
                date_list.append(review_date)
            if reviews2:
                pageno = 3
                while pageno > 2:
                    amazon_url3 = f'https://www.amazon.in/product-reviews/{product_id}/ref=cm_cr_getr_d_paging_btm_next_{pageno}?pageNumber={pageno}'
                    driver.get(amazon_url3)

                
                    for i in range(5):
                        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
                        time.sleep(2)  

                
                    page_source3 = driver.page_source

                
                    soup3 = BeautifulSoup(page_source3, 'html.parser')
                    # print(soup3)
                    print('-'*50)
                    reviews3 = soup3.find_all('span', attrs={'data-hook': 'review-body'})
                    profile_names3 = soup3.find_all('div', class_='a-profile-content')
                    rating = soup3.find_all('span', class_='a-icon-alt')
                    verfication = soup3.find_all('span', class_='a-size-mini a-color-state a-text-bold')
                    dates = soup3.find_all('span', class_='a-size-base a-color-secondary review-date')
                    for review in reviews3:
                    
                        inner_elements = review.find_all('span')

                        for inner_element in inner_elements:
                            review_list.append(inner_element.get_text(strip=True))
                    filtered_reviews = [review for review in review_list if review]
                    filtered_reviews = [review for review in filtered_reviews if is_english(review)]
                    # print(f'Next page reviews: \n' , filtered_reviews)

                    for name in profile_names3:
            
                        user_names = name.find_all('span')

                        for user_name in user_names:
                            users_list.append(user_name.get_text(strip=True))
                    
                    for star in rating:
                        match = re.search(r'(\d+\.\d+)', str(star))
                        if match:
                            rating = float(match.group())
                        rating_list.append(rating)
                    print("rating",rating_list)
                    for verifi in verfication:
                        varification_list.append(verifi.get_text(strip=True))
                    print("varifi",varification_list)
                    for date in dates:
                        # print(date)
                        date = str(date)
                        pattern = r'Reviewed in India on (\d{1,2} \w+ \d{4})'

                        match = re.search(pattern, date)

                        review_date = match.group(1)
                        review_date = datetime.strptime(review_date, "%d %B %Y").date()
                        # print(review_date)
                        date_list.append(review_date)
                    # print("names",users_list)
                    next_page = soup3.find_all('li', class_='a-disabled a-last')

                    
                    # print("names",users_list)
                    next_page = soup3.find_all('li', class_='a-disabled a-last')
                    x = len(next_page)
                    if x != 0:
                        break
                    else:
                        pageno += 1

         
                
            else:
                print('reviews done..')
                
        else:
            print('reviews empty')
            # process.insert(END,f'\n Reviews: {filtered_reviews}')
                         
        
      


    finally:
        # Close the ChromeDriver
        driver.quit()
    sum = 0
    for i in range(len(rating_list)):
        sum = sum + rating_list[i]
    
    
    print(sum)
    avg = sum/len(rating_list)
    print(avg)
    


    if avg >=3 :
        popular = f'The product is popular among customers'
        # messagebox.showwarning('Popular',f'The product is popular among customers')
    else:
         popular = f'The product is not popular among customers'
        # messagebox.showwarning('Popular',f'The product is not popular among customers')
    
    x = set(users_list)
    length_list = []
    for i in range(len(filtered_reviews)):
        length = len(filtered_reviews[i])
        length_list.append(length)

    file_path = 'Recommends/static/Dataset/reviews.csv'
    with open(file_path, 'w', encoding='utf-8', newline='') as csv_file:
        csv_writer = csv.writer(csv_file) 
        csv_writer.writerow(['review', 'user',"verification","date","length"])  
        data_rows = list(zip(filtered_reviews, users_list,varification_list,date_list,length_list))
        csv_writer.writerows(data_rows)

    print(f'The reviews have been saved to {file_path}.')
 



    rv = pd.read_csv('Recommends/static/Dataset/reviews.csv')
    if rv['review'].empty:
        # messagebox.showwarning('Reviews','No reviews found!!...')
        respond_review = 'No reviews found!!'
    else:
        print('Data: ',rv.head())
        print('Total reviews -> ',rv.shape)
        duplicate_mask = rv['user'].duplicated(keep=False)
        rv = rv[~duplicate_mask]
        rv['date'] = pd.to_datetime(rv['date'])
        rv = rv[rv['verification'] == 'Verified Purchase']
        rv = rv.sort_values(by='date',ascending=False)
        print(rv.head())
        
        print("///////////////////////////////////////////////////////////////////////")
        rv['review'] = rv['review'].apply(lambda x:x.lower())
        for i in range(len(rv)):
            lw=[]
            for j in rv['review'].iloc[i].split():
                if len(j)>=3:
                    lw.append(j)
            rv['review'].iloc[i]=" ".join(lw)

        ps = list(";$?.:-()[]/\'_!,")
        rv['review'] = rv['review']

        for p in ps:
            rv['review'] = rv['review'].str.replace(p, '')

        rv['review'] = rv['review'].str.replace(r'\d+', '')
        rv['review'] = rv['review'].str.replace("    ", " ")
        rv['review'] = rv['review'].str.replace('"', '')
        rv['review'] = rv['review'].apply(lambda x: x.replace('\t', ' '))
        rv['review'] = rv['review'].str.replace("'s", "")
        rv['review'] = rv['review'].apply(lambda x: x.replace('\n', ' '))

        rv['review'] = rv['review'].apply(lambda x: x.lower())
        print('Cleaned review: \n',rv['review'])

        stop = set(STOPWORDS)

        def remove_stopwords(review):
            words = review.split()  
            filtered_words = [word for word in words if word.lower() not in stop]
            return ' '.join(filtered_words)

        rv['review'] = rv['review'].apply(remove_stopwords)

        wl = WordNetLemmatizer()

        nr = len(rv)
        lis = []
        for r in range(0, nr):
            ll = []
            t = rv['review'].iloc[r]
            tw = str(t).split(" ")
            for w in tw:
                ll.append(wl.lemmatize(w, pos="v"))
            lt = " ".join(ll)
            lis.append(lt)

        # print(rv.head())
        rv['review'] = lis
        print('Head: ',rv.head())
        X = rv.review.values

        tokenizer = Tokenizer(num_words=1000)
        tokenizer.fit_on_texts(X)

        X = tokenizer.texts_to_sequences(X)
        max_len =  1000
        X = pad_sequences(X, maxlen=max_len, padding='post')
        # print('X: ',X)
        
        pred = model.predict(X)
        pred_list = pred


        pred_array = np.array(pred_list)
        avg_pred = np.mean(pred_array)
        threshold = 0.45


        indiamart_product= product.replace(' ', '%20')
        indiamart_product

        flipkart_product = product.replace(' ','%20')
        flipkart_product      


        c2_url = generate_c2_url(flipkart_product)
        # print('c2_url: ',c2_url)


        c2_soup = scrape_and_get_price(c2_url)
        


        price2 = c2_soup.find('div',attrs={'class':'_30jeq3'}).text



        c3_url = generate_c3_url(indiamart_product)
        # print('c3_url: ',c3_url)

        c3_soup = scrape_and_get_price(c3_url)
        
        price3 = c3_soup.find('p',attrs={'class':'prc fw7 fs16 clr1'}).text
        print('-'*100)
        print('Flipkart: ',price2)
        print('IndiaMart: ',price3)

        price2_value = int(price2.replace('₹', '').replace(',', ''))
        price3_value = int(price3.replace('₹', '').replace(',', ''))


        # Compare the prices

        price = []
        price.append(price2_value)
        price.append(price3_value)

        # Linear Search Algorithm
        def linear_search_algo(arr):

            if not arr:
                return None  

            low_price = arr[0] 

            for price in arr:
                if price < low_price:
                    low_price = price
            if low_price == price2_value:
                    platform = 'Flipkart'
            else:
                    platform = 'IndiaMart'

            return low_price , platform


        lowest_price, platform = linear_search_algo(price)
        if lowest_price >= int(min_) and lowest_price < int(max_):
            prange = f'The product is in the price range'
            pri_ran = 1
            # messagebox.showwarning('Pricing',f'The product is in the price range ')
        else:
            prange = f'The product is not in the price range'
            pri_ran = 0
            # messagebox.showwarning('Pricing',f'The product is not the price range ')
        
        print(f'The lowest price is ₹{lowest_price} on {platform}')

        rev_rat = avg
        pri_range = pri_ran
        sentiscore = avg_pred
        verified = 1
        pro_popu = avg
        wrev_rat = 0.3
        wpri_range = 0.2
        wsentiscore = 0.2
        wverified = 0.1
        wpro_popu = 0.2
        recom_score = (wrev_rat*rev_rat) + (wpri_range*pri_range) + (wsentiscore*sentiscore) +(wverified*verified) + (wpro_popu*pro_popu)

        Score = (recom_score - 0) / (1-0)  # normalizing

        if Score >= threshold:           
            # messagebox.showinfo('RESULT','Good Product')
            prediction = 'Good Product'
        else:
            # messagebox.showinfo('RESULT','Bad Product')           
            prediction = 'Bad Product'
            print(f'Average Value: {Score}')

        print(f'Result: {prediction}')

        # if avg_pred >= threshold:           
        #     # messagebox.showinfo('RESULT','Good Product')
        #     prediction = 'Good Product'
        # else:
        #     # messagebox.showinfo('RESULT','Bad Product')           
        #     prediction = 'Bad Product'
        #     print(f'Average Value: {avg_pred}')

        # print(f'Result: {prediction}')

        # messagebox.showwarning('Reviews','No reviews found!!...'

    elapsed_time = time.time() - start
    print('Elapsed time: ',elapsed_time)

    return prediction, prange, popular, lowest_price, platform


@never_cache
@csrf_exempt
def scrape_products(request):
    if 'user' in request.session:
        global product1, product2, predict1, predict2, range1, range2, popular1, popular2, price1, price2, site1, site2
        product1= request.POST.get('product1')
        pid1 = request.POST.get('pid1')
        product2= request.POST.get('product2')
        pid2 = request.POST.get('pid2')

        min1 = request.POST.get('min1')
        min2 = request.POST.get('min2')
        max1 = request.POST.get('max1')
        max2 = request.POST.get('max2')

        print('Inputs->', product1,pid1,min1,max1,product2,pid2,min2)

        print('Length1 -> ', len(product1),'Length2 -> ',len(product2))

        if not product1 and not product2:
            status = 'failure'
        else:        

            if product1 and pid1 and max1 and min1:
                result1 = input(product1,pid1,min1,max1)
                print(result1)

                predict1,range1,popular1,price1,site1 = result1
                status = 'success'
            else:
                product1 = 'Product One'
                predict1 = ''
                range1 = ''
                popular1 = ''
                price1 = ''
                site1 = ''
                status = 'success'



            if product2 and pid2 and max2 and min2:
                result2 = input(product2,pid2,min2,max2)
                print(result2)

                predict2,range2,popular2,price2,site2 = result2
                status = 'success'
            else:
                product2 = 'Product Two'
                predict2 = ''
                range2 = ''
                popular2 = ''
                price2 = ''
                site2 = ''
                status = 'success'


        print('Status ->', status)


        print('Product one results -> ',predict1,range1,popular1,price1,site1)
        print('Product two results -> ',predict2,range2,popular2,price2,site2)


        data = {
            'p1':product1,
            'p2':product2,
            # 'r1':result1,
            # 'r2':result2,
            'predict1':predict1,
            'range1':range1,
            'popular1':popular1,
            'price1':price1,
            'site1':site1,
            'predict2':predict2,
            'range2':range2,
            'popular2':popular2,
            'price2':price2,
            'site2':site2,
            'status':status,
        }

        print('Combined data - >',data)
        return JsonResponse(data)

        return render(request,'results.html',{'data':data})
    

@never_cache  
def result_page(request):
    if 'user' in request.session:
        global product1, product2, predict1, predict2, range1, range2, popular1, popular2, price1, price2, site1, site2


        print('in resultpage()')
        print('Product one results -> ',predict1,range1,popular1,price1,site1)
        print('Product two results -> ',predict2,range2,popular2,price2,site2)

        data={
            'p1':product1,
            'p2':product2,
            # 'r1':result1,
            # 'r2':result2,
            'predict1':predict1,
            'range1':range1,
            'popular1':popular1,
            'price1':price1,
            'site1':site1,
            'predict2':predict2,
            'range2':range2,
            'popular2':popular2,
            'price2':price2,
            'site2':site2,
        }
        return render(request,'results.html',{'data':data})

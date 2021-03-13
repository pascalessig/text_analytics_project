#!/usr/bin/env python
# coding: utf-8

# # Part 1: Crawler

# This part shows the python script that obtains app data from the google play store. There are multiple helper functions involved and two main function. One crawling an app's metadata, the other crawling reviews that correspond to an app. 
# <br> <br>
# A crawler needs to unite different disciplines in order to fully function. These are (1) connecting to the internet, (2) routing to the target page, (3) modifying the target page in order to make the intormation visible, (4) extracting the HTML code of the target page, (5) searching the HTML file for and extracting the target information, (6) saving the extracted information in a suitable format. 
# <br> <br>
# In order to work through the six mentioned steps, predefined python packages and drivers were used. As driver, the chromedriver compatible with GoogleChrome_v_83.0.4103 was used. The ChromeDriver allows to create a driver object, which can be used to access the internet through the Google Chrome browser, as well as navigating to pages and manipulating its content. The Chrome driver was used in combination with the Selenium webdriver, which allows to access the above mentioned functionality of the Chrome driver using a python script. 
# <br> <br>
# Once the HTML source code is downloaded by the webdriver, the code has to be searched for the target information. To do so, the well known BeautufulSoup python package is used. It deliveres the functionality to efficiently search through HTML code by specifying the type and attributes of HTML elements. It also allows to navigate from a specific HTML element using parent and child elements. 

# In[2]:


# Import
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
import time
from bs4 import BeautifulSoup
import re
import pandas as pd
import datetime
import ast
import time


# In[3]:


# Define Browser Options
chrome_options = Options()
chrome_options.add_argument("--headless") # Hides the browser window
chrome_options.add_experimental_option('prefs', {'intl.accept_languages': 'en,en_US'})


# assign globally used variables.
PATH_chromedriver = r'/usr/local/bin/chromedriver'
PATH_output_dir = '../Data.nosync/'


# ##### Define Webdriver functions to manipulate the webpage.
# Functions allow us to: 
# - click "Show More Button" when crawling reviews
# - scroll down till end of page

# In[4]:


def click_show_more_button(driver):
    """
    Clicks the "show more" button when scrolling down the review page of a play store app
    Tested yet with review pages only. 
    
    Argument: 
    driver -- the selenium driver holding the current session
    """
    time.sleep(1)
    show_more_button = driver.find_elements_by_xpath("//span[@class='RveJvd snByac']")[0]
    driver.execute_script("arguments[0].click();", show_more_button)


# In[5]:


def scroll_down_till_limit(driver, x_times = 1, button = ""):
    """
    Scrolls down a webpage x_times times. Clicking button button when scrolled down if
    specified.
    
    Arguments: 
    driver -- chrome selenium driver of current session
    x_times -- how many times the driver should scroll until the end of the page
    button -- which button to click when scrolled down
    
    Returns: 
    driver -- the webdriver at the final state
    """
    # Scroll page to load whole content
    last_height = driver.execute_script("return document.body.scrollHeight")
    for i in range(0, x_times-1):
        while True:
            # Scroll down to the bottom.
            #driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            driver.execute_script("window.scrollTo(0,document.body.scrollHeight);")
            # Wait to load the page
            time.sleep(1)
            # Calculate new scroll height and compare with last height.
            new_height = driver.execute_script("return document.body.scrollHeight")
            if new_height == last_height:
                break
            last_height = new_height
        if button == "show_more":
            click_show_more_button(driver)
        while True:
            # Scroll down to the bottom.
            #driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            driver.execute_script("window.scrollTo(0,document.body.scrollHeight);")
            # Wait to load the page
            time.sleep(2)
            # Calculate new scroll height and compare with last height.
            new_height = driver.execute_script("return document.body.scrollHeight")
            if new_height == last_height:
                break
            last_height = new_height
    return driver


# ##### Information Retrieval Helper Functions
# These functions allow us to: 
# - return the last part of a link separated by "/". 
# - return the href attribut of a link element in HTML code
# - retrieve HTML code from webpage
# - search for and return specified information contained in the HTML code
# - ensure the absence of "Error 404" page for found error

# In[6]:


def get_last_part_of_link(link):
    """
    Returns the last part of a link which are the characters after the last "/".
    E.g. returns "article" in "www.nyt.com/catalog/article".
    But fitted on playstore category links.
    
    Arguments: 
    link -- category link google play store
    
    Returns:
    string -- created unique category name
    """
    pattern = '.*/([A-Za-z_-]*)(\?.+=([A-Za-z0-9_-]*))?$'
    last_part = re.search(pattern, link)
    if last_part.group(1) != None:
        if last_part.group(3) != None:
            return last_part.group(1).lower() + '_' + last_part.group(3).lower()
        return last_part.group(1).lower()
    else:
        raise ValueError("Make sure Argument is playstore category link.")


# In[7]:


def get_links_by_dict(html, d):
    """
    Returns a list of links, which were found in html and have the specifications 
    specified in dictionary d.
    
    Arguments: 
    html -- html text to search for links in
    d -- dictionary specifying the attribute, value relation, {"attr" : "value"}
    
    Returns: 
    links -- the links in the html file fitting the pattern in d
    """
    assert type(d) == dict, "Argument d has to be of type dictionary"
    
    # Parse HTML structure
    soup = BeautifulSoup(html, "lxml")

    links = []
    #for link_object in soup.find_all("a", d):
    for link_object in soup.find_all("a", d):
        links.append(str(link_object.get('href')))

    return links         


# In[8]:


def get_htmltext(partial_link, scroll=False, prefix = True, postfix=True, get_reviews = False, amount_scroll = 1):
    """
    Get html text from partial play store link. 
    
    Arguments: 
    partial_link -- link fragment leading to the play store page of interest
    scroll -- enable selenium scolling to the bottom of the page before getting html code
    prefix -- using the prefix defined in link_prefix
    postfix -- using the postfix defined in link_postfix
    get_reviews -- boolean whether to get html from review page
    amount_scroll -- specifies how many times the "show_more" button needs to be clicked
    
    Returns: 
    htmltext -- the html code of the current session as string
    """
    
    link_prefix = r"https://play.google.com"
    link_postfix = r"?&hl=en"
    link_get_reviews = r"&showAllReviews=true"

    driver = webdriver.Chrome(executable_path=PATH_chromedriver, options=chrome_options)
    # Run the Webdriver, save page an quit browser
    link = str(partial_link)
    if prefix:
        link = link_prefix + link
    if postfix: 
        link += link_postfix
    if get_reviews:
        link += link_get_reviews
    print(link)
    driver.get(link)
    if scroll:
        if get_reviews: 
            driver = scroll_down_till_limit(driver, x_times = amount_scroll, button = 'show_more')
        driver = scroll_down_till_limit(driver)
    htmltext = driver.page_source
    driver.quit()
    return htmltext


# In[9]:


def get_app_dict(html):
    """
    Returns dictionary containing the metadata on an app page in the play store.
    
    Arguments: 
    html -- html code of the app page 
    
    Returns: 
    app_dict -- dictionary containing the app data
    """
    # parse the hrml code to a bs4 object
    soup = BeautifulSoup(html, "lxml")
    # initialize the app dict
    app_dict = dict()
    # get title of app
    title = list()
    for h1 in soup.find_all("h1"):
        title.append(h1.get_text())
    app_dict['title'] = title[0]

    # get download numbers of app
    downloads = []
    download_pattern = r'.*aria-label=".*ratings".*'
    for span in soup.find_all("span"):
        if re.match(download_pattern, str(span)) != None: 
            downloads.append(str(span.get_text()))
    if len(downloads) > 0:
        app_dict['downloads'] = int(downloads[0].replace(',', ''))

    # get Genre of app
    genre = list()
    for link in soup.find_all("a", itemprop="genre"):
        genre.append(link.get_text())
    if len(genre) > 0:
        app_dict['genre'] = genre[0]

    # get rating of app
    rating_pattern = r'.*aria-label="Rated +* stars out of five start".*'
    rating = -1
    for div in soup.find_all("div", "BHMmbe"):
            rating = float(div.get_text())
    app_dict['rating'] = rating    
    
    # get company details
    comp_name = soup.find('a', {'class':'hrTbp R8zArc'}).text
    comp_link = soup.find('a', {'class':'hrTbp R8zArc'}).get('href')
    app_dict['comp_playstore_link'] = comp_link
    app_dict['comp_name'] = comp_name
    
    # get description of app
    desc = soup.find_all(attrs={"jsname":"sngebd"})
    app_dict['description'] = desc[0].get_text()

    add_info = list()
    values = list()
    for div in soup.find_all("div", "hAyfc"):
        values.append(div.find_all("div", "IQ1z0d")[0])
        topic = div.find_all("div", "BgcNfc")
        add_info.append(topic[0].get_text())

    for topic, value in zip(add_info, values):
        if re.match(r'.*mailto:.*', str(value)) == None:
            app_dict[str(topic)] = value.get_text()
        else:
            for link in value.find_all("a"):
                if re.match(r'.*Visit website.*', str(link)) != None:
                    app_dict['url'] = str(link.get('href'))
                if re.match(r'.*mailto:.*', str(link)) != None: 
                    app_dict['email'] = str(link.get('href'))
            info =value.find_all("div")[-1]
            app_dict['information'] = info.get_text()

    return app_dict


# In[10]:


def check_page(htmltext):
    """
    Check whether google play store page does not exist.
    Error 404.
    Returns False if page does not exist.
    
    Arguments: 
    htmltext -- html text of website
    
    Returns: 
    boolean -- specifying whether page exists or not
    """
    soup = BeautifulSoup(htmltext, "lxml")
    pattern = r".*the requested URL was not found on this server.*"
    for obj in soup.find_all('div', 'uaxL4e', "error-section"):
        if(re.match(pattern, obj.get_text()) != None):
            return False
    return True


# ##### Data Management Helper Functions build with Pandas
# These functions allow us to: 
# - convert a dictionary to a dataframe
# - convert a dataframe to a dictionary
# - save a dictionary as csv file 
# - update a csv file with data

# In[11]:


def get_df_from_dict(d, first_col_name = 'id'):
    """
    Transforms a dictionary of shape {'a' : {'b':'bi', ..., 'z':'zi'}} 
    into a Dataframe. First colname has to be specified otherwise index. 
    
    Arguments: 
    d -- dictionary to convert into Dataframe
    first_col_name -- string specifying the name of the first column
    
    Returns: 
    df -- DataFrame from dict d
    """
    df = pd.DataFrame.from_dict(d, orient='index')
    df.reset_index(inplace=True)
    df.columns = df.columns[1:].insert(0, first_col_name)
    return df


# In[12]:


def save_dict(d, file_name, first_col_name = 'id'):
    """
    Saves dictionary as csv file to disk.

    Arguments:
    d -- dictionary with app data
    file_name -- the file name the data will be saved to.
    first_col_name -- string specifying the name of the first column
    """
    df = get_df_from_dict(d, first_col_name)
    PATH = PATH_output_dir + file_name + '.csv'
    df.to_csv(PATH, index=False)


# In[13]:


def update_csv(PATH_to_csv, d, first_col_name, output_path = ''):
    """
    Updates csv file with data of dictionary.

    Arguments:
    PATH_to_csv -- path of the old data that should be updated
    d -- dictionary with additional app data
    first_col_name -- string specifying the name of the first column
    output_path -- the path the file will be saved to if other than the original files path.
    """
    if output_path == '':
        output_path = PATH_to_csv
    df_old = pd.read_csv(PATH_to_csv)
    df_new = get_df_from_dict(d, first_col_name)
    df = pd.concat([df_old, df_new], axis = 0, ignore_index = True)
    df.to_csv(output_path, index=False)


# In[14]:


def recover_dict_from_csv(PATH_to_csv, convert = {}):
    """
    Transforms a pandas dataframe with two columns to a dictionary 
    {col1_1 : col2_1, ..., col1_n : col2_n}
    
    Arguments: 
    PATH_to_csv -- the path leading to a csv with two columns
    convert -- converters to consider when loading csv in DataFrame
    
    Returns:
    d -- dictionary at form specified above
    """     
    
    try:
        df = pd.read_csv(PATH_to_csv, converters=convert)
        d = df.set_index(df.columns[0]).to_dict('index')
    except: 
        raise ValueError('no .csv available at path!')
    
    return d
        


# ##### Functionalities Helper Functions
# The following functions help us to: 
# - find the URLs poining to app collections allocated in categories
# - find the URLs poining to app cluster pages within category pages

# In[15]:


def get_cat_dict(main_page_id, main_page_url):
    """
    Returns a dictionary containing cat_id : {cat_name : category_name, link : category_partial_link, clusters : cluster_list} pairs. 
    
    Arguments: 
    main_page_id -- id of main page e.g. 01 for google play store
    main_page_url -- url of main page
    
    Returns: 
    cat_dict -- category dictionary of form as specified above
    """

    main_page_html = get_htmltext(partial_link=main_page_url, scroll=False, postfix=True, prefix=False)
    cat_links = get_links_by_dict(main_page_html, d = {'class' : 'r2Osbf'})
    cat_names = [get_last_part_of_link(link) for link in cat_links]
    cat_ids = [main_page_id + '_' + str(i).zfill(2) for i in range(1,len(cat_names)+1)]
    clusters = [get_cluster_dict(cat_link) for cat_link in cat_links]
    cat_dict = {cat_ids[i]: {'cat_name' : cat_names[i], 'cat_link' : cat_links[i], 'clusters' : clusters[i]} for i in range(0,len(cat_names))}
    return cat_dict


# In[16]:


def get_cluster_dict(cat_link):
    """
    Returns a list containing category cluster partial links . 
    
    Arguments: 
    cat_dict -- a category dictionary that contains links for sites where to search cluster links in
    
    Returns: 
    cluster_links -- list of links of app clusters
    """        
    html = get_htmltext(cat_link, scroll=True, postfix=False)
    cluster_links = get_links_by_dict(html, d={'class' : 'LkLjZd ScJHi U8Ww7d xjAeve nMZKrb id-track-click'})
    if len(cluster_links) == 0:
        return [cat_link]
    else:
        return list(set(cluster_links)) # prevent link duplicates


# ##### Main Funcitons
# The following functions: 
# - collect the app metadata given an app dictionary
# - collect review data of an app given an app dictionary
# - split the total link list poiting to apps in 5 equally sized lists to speed up crawling

# In[17]:


def get_app_link_dict(cat_dict = ''):
    """
    Returns a dictionary containing {app_id : app_link}
    If argument cat_links not specified, the disk is searched for a csv file containing the cat_links / cluster_links
    
    Arguments: 
    cat_dict -- if specified the category links are taken from this dict
    
    Returns: 
    app_link_dict -- dict containing app_links as specified above.
    """
    
    # Get htmls of cluster pages
    if cat_dict == '':
        cat_dict = recover_dict_from_csv('/Users/martinthoma/Desktop/Projects/sophia_awesome/play_store_crawler/category_master.csv', convert={'clusters' : lambda x: ast.literal_eval(x)})

    app_link_dict = dict()
    count = 0
    a = len(list(cat_dict.keys()))
    
    for cat_id, d in cat_dict.items():
        app_links_in_cat = list()
        for link in d['clusters']:
            print('-------------')
            html = get_htmltext(link, scroll=True, postfix=False)
            if not check_page(html):
                print("Faulty Page! ")
            app_links_in_cat += get_links_by_dict(html, d = {'class' : 'poRVub'})
        app_links_in_cat= list(set(app_links_in_cat))    
        for i in range(0,len(app_links_in_cat)):
            app_id = cat_id + '_' + str(i+1).zfill(4)
            app_link = app_links_in_cat[i]
            app_link_dict[app_id] = {'app_link' : app_link}
        print(len(app_link_dict))        
        count+=1
        print('%i of %i cluster htmls downloaded.'%(count, a))
    return app_link_dict


# In[18]:


def get_rev_dict(app_id, html, log_dict):
    """
    Searches html of revies page of app in google play store  for reviews. 
    Puts reviews in dictionary. 
    
    Arguments: 
    app_id -- app_id of the current app
    html -- html text of the reviews page of the current app
    log_dict -- dicrionary conatining the logs for incidents
    
    Returns:
    amount_revs -- the amount of reviews collected from the current application
    rev_dict -- dictionary containing the reviews data of the current app
    log_dict actuallized dicrionary conatining the logs for incidents
    """
    soup = BeautifulSoup(html, "lxml")
    rev_el = soup.find_all("div", {"class": "d15Mdf bAhLNe"}) 
    rev_dict = dict()
    amount_revs = len(rev_el)
    counter = 1
    for el in rev_el:
        try:
            rev_id = app_id + '_' + str(counter).zfill(4)
            rev_author = el.find('span', {"class": "X43Kjb"}).text
            rev_date = str(datetime.datetime.strptime(el.find('span', {"class": "p2TkOb"}).text, '%B %d, %Y').date())
            rev_rating = el.find('div', {"role": "img"})['aria-label'][6]
            rev_cnt_helpful = el.find('div', {"aria-label": "Number of times this review was rated helpful"}).text
            rev_text = el.find('span', {"jsname": "fbQN7e"}).text
            if len(rev_text) == 0: 
                rev_text = el.find('span', {"jsname": "bN97Pc"}).text
        except:
            log_dict[rev_id] = {'log' : 'Error reading review data'}
            
        rev_dict[rev_id] = {'author': rev_author,
                            'date' : rev_date,
                            'rating' : rev_rating,
                            'cnt_helpful' : rev_cnt_helpful,
                            'text' : rev_text}
    
        counter +=1
    return amount_revs, rev_dict, log_dict


# In[19]:


def get_reviews_dict(PATH_to_csv, filename):
    """
    Arguments: 
    PATH_to_csv -- path to csv file containing app links, app_link_master.csv
    filename -- name of the csv file to save the reviews in
    
    Returns: 
    reviews_dict -- dictionary containing the review data of all apps in the PATH_to_csv file
    log_dict -- dicrionary conatining the logs for incidents
    """
    #'/Users/martinthoma/Desktop/Projects/sophia_awesome/play_store_crawler/app_link_master.csv'
    app_link_dict = recover_dict_from_csv(PATH_to_csv)
    total = len(app_link_dict.keys())
    total_revs, count, tic = (0, 1, time.time())
    reviews_dict, log_dict = dict(), dict()
    for app_id, d in app_link_dict.items():
        try: 
            html = get_htmltext(d['app_link'], scroll=True, postfix=False, get_reviews=True, amount_scroll=2)
        except:
            try:
                html = get_htmltext(d['app_link'], scroll=True, postfix=False, get_reviews=True, amount_scroll=1)
            except:
                log_dict[app_id] = {'log' : 'Error while reading HTML code'}
                continue;
        if check_page(html):
            amount_revs, rev_dict, n_log_dict = get_rev_dict(app_id, html, log_dict)            
            reviews_dict.update(rev_dict)
            log_dict = n_log_dict
            progress = (count / total)
            eta = ((time.time() - tic) / progress / 3600) * (1-progress)
            total_revs += amount_revs
            print('Progress: ' + str(round(progress*100, 2)) + '%,\tETA: ' + str(round(eta,2)) + 'h\tTotal Reviews: ' + str(total_revs))
            print('--------------------')
            count += 1
            if (count % 100 == 0):
                save_dict(reviews_dict, filename, first_col_name='app_id')
    save_dict(reviews_dict, filename, first_col_name='rev_id')
    return reviews_dict, log_dict


# In[20]:


def get_apps_dict(PATH_to_csv, filename):
    """
    Arguments: 
    PATH_to_csv -- path to csv file containing app links, app_link_master.csv
    filename -- name of the csv file to save the app_data in
    
    Returns: 
    apps_dict -- dictionary containing the meta data of all apps in the PATH_to_csv file
    log_dict -- dicrionary conatining the logs for incidents
    """
    #'/Users/martinthoma/Desktop/Projects/sophia_awesome/play_store_crawler/app_link_master.csv'
    app_link_dict = recover_dict_from_csv(PATH_to_csv)
    total, count, tic = (len(app_link_dict.keys()), 1, time.time())
    apps_dict, log_dict = dict(), dict()

    for app_id, d in app_link_dict.items():
        try: 
            html = get_htmltext(d['app_link'], scroll=False, postfix=False, get_reviews=False)
        except:
            log_dict[app_id] = {'log' : 'Error while reading HTML code'}
            continue;
        
        try:
            app_dict = get_app_dict(html)
        except:
            log_dict[app_id] = {'log' : 'Error while extracting app data'}
            continue;
            
        app_dict['play_store_link'] = d['app_link']
        apps_dict[app_id] = app_dict
        
        progress = (count / total)
        print('Progress: ' + str(round(progress*100, 2)) + '%')
        toc = time.time()
        eta = ((time.time() - tic) / progress / 3600) * (1-progress)
        print('Progress: ' + str(round(progress*100, 2)) + '%,\tETA: ' + str(round(eta,2)) + 'h')
        print('--------------------')
        count += 1
        if (count % 100 == 0):
            save_dict(apps_dict, filename, first_col_name='app_id')
    save_dict(apps_dict, filename, first_col_name='app_id')
    return apps_dict, log_dict
    


# In[21]:


def split_data_in_five():
    """
    Splits the found app links in five equally sized and hardcoded datasets.
    This allows to paralellize the task at hand, leading to the reduction of 
    crawling time to about 20% of the original crawling time. 
    """
    # split app links to make paralizable
    # not finished yet
    df = pd.read_csv('/Users/martinthoma/Desktop/Projects/sophia_awesome/play_store_crawler/app_link_master.csv')

    df[:1183].to_csv('app_links_A.csv', index=False)
    df[1183:2366].to_csv('app_links_B.csv', index=False)
    df[2366:3549].to_csv('app_links_C.csv', index=False)
    df[3549:4732].to_csv('app_links_D.csv', index=False)
    df[4732:].to_csv('app_links_E.csv', index=False)


# # Part 2: Cleaning Data
# 
# Directly after downloading, the app metadata is not ready for the task specific preprocessing and has to be cleaned. To do so, the following helper functions were used. The functionallity of each function is described in the functions description text. 
# <br><br>
# The data_cleaning() function is the main function which uses the helper functions to clean the data obtaines by the crawler. 

# In[22]:


# imports 
import glob
import pandas as pd
import numpy as np
import re


# In[23]:


PATH_output_exam_project = '../Data.nosync/app_data_exam_09052020.csv'
PATH = '../Data.nosync/*.csv'


# In[24]:


def get_app_df(PATH):
    """
    Arguments: 
    PATH -- path to the app files. Has to end with common_part_of_name*.csv
    
    Returns: 
    df -- dataframe containing all app data unioned
    """
    filenames = glob.glob(PATH)
    dfs = list()
    for filename in filenames:
        dfs.append(pd.read_csv(filename, parse_dates=['Updated']))
    df = pd.concat(dfs, axis=0, join='outer', ignore_index=True)
    return df


# In[25]:


def to_float_or_nan(x): 
    """
    Used with anonymous lambda function and pandas dataframe
    
    Arguments: 
    x -- Element of pandas series 
    
    Returns: 
    -- float value of x if conversion possible
    -- NaN if conversion not possible
    """
    try:
        return float(x)
    except: 
        return np.nan


# In[26]:


def remove_mailto(x):
    """
    Removes the mailto: string in front of an email. 
    Used with anonymous lambda function and pandas dataframe
    
    Arguments: 
    x -- Element of pandas series 
    
    Returns: 
    email -- plain email string 
    Unknown -- if email can not be retrieved
    """
    pattern = r'^(mailto:)(.+@[\w-]+.\w{2,3})'
    match = re.match(pattern, x)
    if match != None: 
        return match.group(2)
    else:
        print('email %s could not be converted to email' %x)
        return 'Unknown'


# In[27]:


def split_in_app_products(x, return_i):
    """
    Splits the price range of the In-app-products column and 
    returns the specified fragment
    Used with anonymous lambda function and pandas dataframe
    
    Arguments: 
    x -- Element of pandas series
    return_i -- Specification of group of re object to be returned
    
    Returns: 
    t[return_i] -- the group of the match object as specified in the arguments.
    """
    assert type(return_i) == int, 'Type of return_i has to be int'
    assert (return_i < 4) & (return_i >= 0), 'return_i has to be between 0 and 3'
    pattern = r'^([$€])((\d{1,2},)?\d{1,3}.\d{1,2})( - ([$€])((\d{1,2},)?\d{1,3}.\d{1,2}))? per item'
    if str(x) == 'nan':
        t = (np.nan, np.nan, 'Unknown', 'Unknown')
        return t[return_i]
    match = re.match(pattern, str(x))
    if match != None: 
        if match.group(6) != None:
            t =  (match.group(2), match.group(6), match.group(1), 'per item')
        else:
            t =  (match.group(2), np.nan, match.group(1), 'per item')
    else:
        print('%s could not be detected as In-app_purchase' %str(x))
        t  = (np.nan, np.nan, 'Unknown', 'Not Detected')
    return t[return_i]


# In[28]:


def requires_android(x):
    """
    Returns the version of android needed for installing an app
    Used with anonymous lambda function and pandas dataframe
    
    Arguments: 
    x -- Element of pandas series
    
    Returns: 
    A string containing the version 
    """
    pattern_version = r'^(\d{1,2}.\d(.\d)?).*'
    pattern_varies = r'Varies with device'
    match_version = re.match(pattern_version, str(x))
    match_varies = re.match(pattern_varies, str(x))
    if match_version != None: 
        return match_version.group(1)
    elif match_varies != None: 
        return match_varies.group(0)
    else:
        return('Unknown')


# In[29]:


def data_cleaning(): 
    """
    Uses the above defined helper functions to work through the data, cleaning it. 
    Finally, the cleaned data is saved in a new csv file. 
    """
    df = get_app_df(PATH_csv)

    df['downloads'] = df['downloads'].apply(lambda x: to_float_or_nan(x))
    df['rating'] = df['rating'].apply(lambda x: to_float_or_nan(x))
    df['email'] = df['email'].apply(lambda x: remove_mailto(x))

    df['In-app Products From'] = df['In-app Products'].apply(lambda x: split_in_app_products(x, 0))
    df['In-app Products To']= df['In-app Products'].apply(lambda x: split_in_app_products(x, 1))
    df['In-app Products Currency'] = df['In-app Products'].apply(lambda x: split_in_app_products(x, 2))
    df['In-app Products type'] = df['In-app Products'].apply(lambda x: split_in_app_products(x, 3))

    df['Requires Android'] = df['Requires Android'].apply(lambda x: requires_android(x))

    # drop unnamed as it contains no information
    # drop Permissions Report and Offered By as they only contain one single value
    # without any information
    df.drop(['Unnamed: 0', 'Permissions', 'Report', 'Offered By', 'In-app Products'], axis=1, inplace=True)
    
    df.to_csv(PATH_output)


# In[ ]:





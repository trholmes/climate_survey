import matplotlib
matplotlib.use('Agg')
import math
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors
from matplotlib.ticker import PercentFormatter
import seaborn as sns

# Choose colors for pie charts
all_colors = sns.color_palette("husl", 40)
#all_colors = sns.color_palette("Set2")

resp_files = {
        "grad":{
        2021: "data/Physics Graduate Survey 2021 (Responses) - Form Responses 1.csv",
        2022: "data/Physics Graduate Survey 2022 (Responses) - Form Responses 1.csv",
        2023: "data/Physics Graduate Survey 2023.csv",
        },
        "ugrad":{
            2021: "data/Physics Undergrad Survey 2021 (Responses) - Form Responses 1.csv",
            2022: "data/Physics Undergrad Survey 2022 (Responses) - Form Responses 1.csv",
            2023: "data/Physics Undergrad Survey 2023.csv",
        }
}

years = [2023, 2022, 2021]
#years = [2023]
population = "ugrad"
dataset = f"{population}_{years[0]}"
output_dir = "plots/"+dataset+"/"

import json
f = open('selections.json')
selections = json.load(f)
#TODO: I didn't actually read the files from here, could do that to make things more streamlined

### HELPER FUNCTIONS ###

# Set up simplified bins
simplified_bins = {
    selections[dataset]["us"]: {"US Education": ["Yes"], "Non-US Education": ["No"],},
    selections[dataset]["year"]: {"2022": [2022], "2023": [2023], "2024": [2024], "2025": [2025], "2026": [2026], "2027": [2027]},
    #selections[dataset]["year"]: {"2023": [2023], "2022": [2022], "2021": [2021], "2020": [2020], "2019 or Earlier": [2019, 2018, 2017, 2016, 2015, 2014]},
    selections[dataset]['gender']: {'Male': ['Male'], 'Female and Nonbinary': ['Female', 'Nonbinary / Third Gender', 'Woman', 'Man', ]},
    selections[dataset]['race']: {'Only selected White/Caucasian': ['White/Caucasian'], 'Other': ['White/Caucasian, Hispanic', 'Pacific Islander, Asian', 'Hispanic', 'White/Caucasian;Native American', 'White/Caucasian, Native American', 'South Asian', 'Asian', 'Black or African American', 'White/Caucasian;Prefer not to disclose', 'White/Caucasian, Asian', 'White/Caucasian, Asian, Native American', 'White/Caucasian, Hispanic, Jewish', 'White/Caucasian;Asian', 'White/Caucasian;Hispanic', 'Black or African American, Hispanic', 'Hispanic, White (latino)', 'Arab', ]},
    selections[dataset]['lgbtq']: {'Only selected Heterosexual': ['Heterosexual/Straight', 'Heterosexual/Straight, never think about this question', 'Straight'], 'Other': ['Asexual', 'Non-identifying', 'Bisexual/Pansexual, Heterosexual/Straight', 'Bisexual/Pansexual;Queer', 'Heterosexual/Straight, Asexual, Questioning', 'Bisexual/Pansexual, Demisexual', 'Heterosexual/Straight;Questioning', 'Bisexual/Pansexual, Gay', 'Queer', 'Heterosexual/Straight;Queer', 'Gay', 'Asexual, Prefer not to disclose', 'Lesbian', 'Straight ', 'Bisexual/Pansexual;Gay', 'Bisexual/Pansexual', 'Bisexual/Pansexual;Heterosexual/Straight', 'Bisexual/Pansexual, Queer', 'Questioning', ]},
    }

# This function is meant to identify the questions that are agree/disagree 1-5
# so that we can make sure all values are listed in the plot and it's in order
def isNumeric(n_q, bin_vals):
    numeric_vals = [1,2,3,4,5,"1","2","3","4","5"]
    for n in numeric_vals:
        #for v in bin_vals: print(v, type(v))
        if n in bin_vals:
            if not "many" in questions[years[0]][n_q]:
                simplified_bins[n_q] = {"Strongly disagree": [1, "1.0"], "Disagree": [2, "2.0"], "Neither agree nor disagree": [3, "3.0"], "Agree": [4, "4.0"], "Strongly agree": [5, "5.0"]}
            return True
    return False

# Get bins for this histogram
def getBins(resps, n_qs, simplified=True):

    n_q = n_qs[years[0]]
    vals = []
    for yr in resps:
        vals.extend(resps[yr].iloc[:, n_qs[yr]])
    vals = np.array(vals)

    #print("TH: getBins()")
    #print(vals)

    #vals = np.array(resps.iloc[:, n_q])

    # Avoid issue with null responses
    cleaned_vals = vals[~pd.isnull(vals)]
    cleaned_vals = np.delete(cleaned_vals, np.where(cleaned_vals == 'nan'))
    unique_vals = np.unique(cleaned_vals)
    #has_nulls = False
    if len(vals) != len(cleaned_vals):
        unique_vals = np.append(unique_vals, "No Response")
        # Make all of the numeric values integer strings
        if not "many" in questions[years[0]][n_q]:
            for tmpval in ["1.0", "2.0", "3.0", "4.0", "5.0"]:
                unique_vals[unique_vals==tmpval] = int(tmpval[0])
        #has_nulls = True

    # If vals are numeric, use the full range
    numeric_vals = [1,2,3,4,5]
    #if has_nulls: numeric_vals = ["1.0", "2.0", "3.0", "4.0", "5.0"]
    #print("TH:", n_q, "unique_vals", unique_vals)
    is_numeric = isNumeric(n_q, unique_vals)
    #print("TH:", n_q, "is numeric?", is_numeric)
    if is_numeric:
        unique_vals = np.unique(np.append(unique_vals, numeric_vals))

    # Option to use hardcoded combinations
    if simplified and n_q in simplified_bins:
        simp_vals = np.array(list(simplified_bins[n_q].keys()))
        if len(vals) != len(cleaned_vals):
            if "No Response" not in simplified_bins[n_q]:
                simp_vals = np.append(simp_vals, "No Response")
                simplified_bins[n_q]["No Response"] = ["No Response"]
        return simp_vals

    # Otherwise return unique list
    return unique_vals

# Simple poisson error
def getErr(arr):
    err_arr = []
    for entry in arr:
        err_arr.append(math.sqrt(entry))
    return err_arr

# Get a title and split it when appropriate. Maybe there is a less dumb way to do this.
def getSplitString(s, max_words = 9):

    # Clean up super long titles
    if s.startswith("To what extent"):
        s = s.split('"')[1]

    n_words = len(s.split())
    n_breaks = int(n_words/max_words)
    if n_words <= max_words: return s

    title = ""
    br = 0
    for br in range(n_breaks):
        title += " ".join(s.split()[br*max_words:(br+1)*max_words])
        title += "\n"
    title += " ".join(s.split()[(br+1)*max_words:])
    return title

# When bin labels are strings, split them as appropriate for display - mainly relevant for pie charts
def getBinLabels(bins, max_words = 5):
    mod_bins = []
    for b in bins:
        if not type(b)==str:
            mod_bins.append(b)
        else:
            mod_bins.append(getSplitString(b, max_words))
    return mod_bins

# Get a histogram and its errors for a given question and selection
def getHistAndErr(resps, vals, n_q, bins, sel="True", normalize=True, simplified=True):

    n_tot = 0
    bin_vals = [0]*len(bins)
    for j, val in enumerate(vals):
        #print("TH:", sel)
        #print(dataset)
        #print(selections[dataset]['gender'])
        #print(resps.iloc[:, selections[dataset]['gender']][j])
        #print(eval(sel))
        if eval(sel):
            comp_val = val

            # Deal with null values
            if pd.isnull(val):
                comp_val = "No Response"
            elif "No Response" in bins and not (simplified and (n_q in simplified_bins)):
                comp_val = str(val)

            # Option to use simplified categories
            if simplified and n_q in simplified_bins:
                for k in simplified_bins[n_q]:
                    if comp_val in simplified_bins[n_q][k]:
                        comp_val = k
                        break

            #print("Current simplified_bins:", simplified_bins)
            #print("n_q:", n_q, "comp_val:", comp_val, "bins:", bins)
            try:
                i = np.where(bins == comp_val)[0][0]
                bin_vals[i] += 1
                n_tot += 1
            except:
                if comp_val != "Prefer not to disclose":
                    print("No entry added in question", n_q, "for value", comp_val)
                    print("\t Possible bins:", bins)
                    print("\t Type of comp_val:", type(comp_val))
    bin_errs = getErr(bin_vals)

    if normalize:
        if n_tot > 0:
            for j in range(len(bin_vals)):
                    bin_vals[j] = bin_vals[j]/n_tot
                    bin_errs[j] = bin_errs[j]/n_tot

    return bin_vals, bin_errs

# Get mean value of distribution -- only valid for 1-5 distributions
def getMean(data, nprofs=False):

    val_index = [1, 2, 3, 4, 5]
    if nprofs: val_index = [0, 1, 2, 3, 4, 5]
    vals = data[:len(val_index)] # Strip off "no response"
    count = 0
    total = 0
    dev_total = 0
    for i, v in enumerate(vals):
        count += v
        total += (val_index[i])*v
    if count>0:
        mean = total/count
        for i, v in enumerate(vals):
            dev_total += v*(val_index[i] - mean) ** 2
        dev = math.sqrt(dev_total / count)
        err = dev / math.sqrt(count)
        return "%.2f ± %.2f"%(mean, err)
    return -1

# Make a plot comparing response values for different selections
def plotData(resps_array, n_q, sels={"all":"True"}, app="", normalize=True, pie=False, simplified=True):

    year = years[0]
    resps = resps_array[year]
    n_qs = {year: n_q}

    if len(resps_array)>1:
        for y in resps_array:
            if y == year: continue
            n_qs[y] = getNQ(resps, resps_array[y], n_q)

    #bins = getBins(resps, n_q, simplified=simplified)
    bins = getBins(resps_array, n_qs, simplified=simplified)
    data = {}
    for sel in sels:
        data[sel] = {}
        vals = np.array(resps.iloc[:, n_q])
        data[sel]["bin_vals"], data[sel]["bin_errs"] = getHistAndErr(resps, vals, n_q, bins, sels[sel], normalize, simplified=simplified)

    datas = {year: data}
    if len(resps_array)>1:
        for y in resps_array:
            if y == year: continue
            if n_qs[y]==-1:
                #print("TH: No question found in year", y)
                #print("\t Question was:", questions[year][n_q])
                continue
            datas[y] = {}
            for sel in sels:
                datas[y][sel] = {}
                vals = np.array(resps_array[y].iloc[:, n_qs[y]])
                datas[y][sel]["bin_vals"], datas[y][sel]["bin_errs"] = getHistAndErr(resps, vals, n_q, bins, sels[sel], normalize, simplified=simplified)

    #TH: Debug
    #print(n_q)
    #print(n_qs)
    #print(bins)

    maxxlabel = 0
    for x in bins: maxxlabel = max(len(str(x)), maxxlabel)

    # Figure out canvas size
    xsize = 5
    ysize = 5
    if maxxlabel > 30:
        xsize = 7
        ysize = 7
    if len(bins)>6:
        xsize = 9

    # Make plot
    fig, axs = plt.subplots(1, 1, figsize=(xsize,ysize))

    if pie:
        if len(sels) != 1:
            print("Cannot make pie chart for multiple selections yet.")
            return
        for sel in sels:
            #axs.pie(data[year][sel]["bin_vals"], explode=explode, labels=getBinLabels(bins), autopct='%1.1f%%', shadow=True, startangle=90)
            #colors = all_colors[:len(bins)]
            colors = getColors(len(bins))
            patches, texts, autotexts = axs.pie(datas[year][sel]["bin_vals"], labels=getBinLabels(bins, 2), autopct='%1.1f%%', textprops={'fontsize': 12}, colors=colors)
            axs.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
            axs.margins(0.5)
            for autotext in autotexts:
                autotext.set_color('white')
            axs.set_title(getSplitString(questions[year][n_q]), fontsize=10)
            #plt.tight_layout()
            plt.savefig(output_dir+"q_%d%s_pie.png"%(n_q, app))
            plt.close()
        return

    for sel in sels:
        for yr in resps_array:
            if n_qs[yr] == -1: continue
            mylabel = sel
            if showMean(bins, n_q):
                mylabel = "%s (mean: %s)"%(sel, getMean(datas[yr][sel]["bin_vals"], nprofs=("How many faculty members" in questions[year][n_q])))
                if len(resps_array)>1: mylabel = mylabel.replace(sel, str(yr))
            elif len(resps_array)>1: mylabel = str(yr)
            axs.errorbar(range(len(bins)), datas[yr][sel]["bin_vals"], yerr=datas[yr][sel]["bin_errs"], label=mylabel, marker="o")

    plt.xticks(range(len(bins)), getBinLabels(bins))
    axs.margins(0.2)
    plt.ylim(bottom=0)
    if normalize: plt.ylabel("Fraction in Group")
    else: plt.ylabel("Number of Respondents")
    if len(questions[year][n_q].split())>30:
        plt.subplots_adjust(top=(1-len(questions[year][n_q].split())*0.005))
    if False: # maxxlabel > 15 and maxxlabel <= 30:
        plt.xticks(rotation=10)
        axs.set_xticklabels(getBinLabels(bins), ha='right')
        plt.subplots_adjust(bottom=maxxlabel*0.007)
    if maxxlabel > 9:
        plt.xticks(rotation=60)
        axs.set_xticklabels(getBinLabels(bins), ha='right')
        plt.subplots_adjust(bottom=min(0.4,maxxlabel*0.03))
    axs.set_title(getSplitString(questions[year][n_q]), fontsize=10)
    if len(sels)>1 or len(resps_array)>1: plt.legend(loc='best', numpoints=1, framealpha=1) #, bbox_to_anchor=(0.5, 1.5))
    else:
        if showMean(bins, n_q): plt.text(0.6, 0.85, "mean: %s"%(getMean(datas[year][sel]["bin_vals"], ("How many faculty members" in questions[year][n_q]))), transform = axs.transAxes)
    plt.savefig(output_dir+"q_%d%s.png"%(n_q, app))
    plt.close()
    return

def showMean(bins, n_q):
    if isNumeric(n_q, bins):
        return True
    if "Disagree" in bins:
        return True
    return False

def getAllSelections(responses, n_q, simplified=True):
    selections = {}
    n_qs = {year: n_q}
    if simplified and n_q in simplified_bins:
        for k in simplified_bins[n_q]:
            selections[k] = "resps.iloc[:, %d][j] in simplified_bins[%d]['%s']"%(n_q, n_q, k)
        return selections

    values = getBins(responses, n_qs)
    for v in values:
        if type(v)==np.int64: # Value is an int
            selections[v] = "resps.iloc[:, %d][j] == %s"%(n_q, v)
        else:
            selections[v] = "resps.iloc[:, %d][j] == '%s'"%(n_q, v)
    return selections

def plotSelections(resps, x, key):
    plotData(resps, x, getAllSelections(resps, selections[dataset][key]),
             app="_"+key)

def getNQ(resp, resp2, n_q):
    quest = resp.columns.values
    quest2 = resp2.columns.values
    key_q = quest[n_q]
    try: return np.where(quest2 == key_q)[0][0]
    except: return -1

def getColors(num):
    my_palette = sns.husl_palette(n_colors=26,s=1, l=0.65)
    selected_colors = []
    for i in range(num):
        selected_colors.append(my_palette[(i*9+1*((i*9)//26))%(26)])
    return selected_colors

# Run Options
dump_questions = False
dump_responses = 0

responses = {}
questions = {}
for yr in years:

    print(f"Working with {population}s from {yr}.")
    with open(resp_files[population][yr], newline='') as csvfile:
        resps = pd.read_csv(csvfile)
        quests = resps.columns.values
    responses[yr] = resps
    questions[yr] = quests

    # Quickly get list of enumerated questions
    if dump_questions:
        for i, q in enumerate(quests):
            print(i, q)
        exit()

    # Get list of all responses for a question
    if dump_responses > 0:
        ans = getBins(resps,dump_responses, False)
        for a in ans:
            print('"%s": "%s",'%(a, a), end = "")
        exit()


# Main plotting:
#myrange = [33]
myrange = range(len(questions[years[0]]))
for x in myrange:

    # Just make a simple plot of everything, y-axis shows counts
    #plotData(responses, x, {"all":"True"}, app="_all", normalize=False, simplified=True)
    # Pie chart version of above
    #plotData(responses, x, {"all":"True"}, app="_allpie", normalize=False, pie=True, simplified=True)

    # To do comparisons by year, turn on normalization just compare distribution shapes
    # You'll also need to put multiple years into the "years" variable
    #plotData(responses, x, {"all":"True"}, app="_comp", normalize=True, simplified=True)

    # Run these for a single year: compares responses from different populations,
    # assuming you've already set up these keywords in selections.json
    #plotSelections(responses, x, "gender")
    #plotSelections(responses, x, "race")
    #plotSelections(responses, x, "lgbtq")
    #plotSelections(responses, x, "us")
    #plotSelections(responses, x, "year")

    # You can also generate one of these on the fly with the getAllSelections function,
    # which will work for ones that you haven't defined in selections.json
    #plotData(responses, x, getAllSelections(responses, selections[dataset]["gender"]), app="_genders")
    #plotData(responses, x, getAllSelections(responses, 97), app="_race")
    #plotData(responses, x, getAllSelections(responses, 101), app="_nationality")
    #plotData(responses, x, getAllSelections(responses, 3), app="_funding")

    #plotData(responses, x, getAllSelections(responses, 56), app="_lgbtq")
    #plotData(responses, x, getAllSelections(responses, 3), app="_years")
    #plotData(responses, x, getAllSelections(responses, 2), app="_transfers")
    #plotData(responses, x, getAllSelections(responses, 8), app="_career")
    #plotData(responses, x, getAllSelections(responses, 58), app="_nontrad")
    #plotData(responses, x, getAllSelections(responses, 53), app="_race")
    continue

# Look at specific populations' experiences over time
# These are a little bit un-tested, so please validate the output if you're using it.
#plotData(responses, 59, {"gender": "resps.iloc[:, 76][j] in simplified_bins[76]['Female and Nonbinary']"}, app="_comp_gender", normalize=True, simplified=True)
#plotData(responses, 64, {"gender": "resps.iloc[:, 76][j] in simplified_bins[76]['Female and Nonbinary']"}, app="_comp_gender", normalize=True, simplified=True)


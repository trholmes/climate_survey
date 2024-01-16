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
            2021: "data/Physics Graduate Survey 2021 (Responses) - Form Responses 1.csv",
            2022: "data/Physics Graduate Survey 2022 (Responses) - Form Responses 1.csv",
            2023: "data/Physics Undergrad Survey 2023.csv",
        }
}

#years = [2023, 2022, 2021]
years = [2023]
population = "grad"
dataset = f"{population}_{years[0]}"
output_dir = "plots/"+dataset+"/"

import json
f = open('selections.json')
selections = json.load(f)

### HELPER FUNCTIONS ###

    # Useful question numbers
    # 53 = race/ethnicity, 54 = gender, 58 = non-traditional students
    # 2 = transfers, 3 = year

# Set up simplified bins
simplified_bins = {
    # 8: {"Wants to go to physics grad school": ["Graduate school in physics or astronomy"], "Other": ['Considering multiple of the above options', 'Government Contractor - Threat Radar Engineering', 'Graduate school in an engineering discipline or computer science', 'Graduate school in another STEM field', 'Military', 'Not sure', 'Other graduate or professional school (medical, law, pharmacy, etc)', 'Other industry, not including Tech (e.g. optics, oil, etc.)', 'Teaching K-12',]},
    #56: {"Heterosexual/Straight": ['Heterosexual/Straight'], "All Others": ['Asexual', 'Asexual, Prefer not to disclose', 'Bisexual/Pansexual', 'Bisexual/Pansexual, Demisexual', 'Bisexual/Pansexual, Queer', 'Prefer not to disclose', 'Queer', 'Questioning']},
    #54: {"Man": ["Man"], "Woman & Other": ["Woman", "Nonbinary / Third Gender", "Prefer not to disclose"]},
    #58: {"Non-Traditional": ["Yes"], "Traditional": ["No"]},
    #11: {'Did not participate in research':['Did not participate in research'],
        #'Combination paid/unpaid':['Research for class credit, Research for pay', 'Research for class credit, Research for pay, Research for a combination of credit and pay', 'Research for class credit, Research for pay, Research for neither credit nor pay (volunteering time)', 'Research for class credit, Research for pay, Research for neither credit nor pay (volunteering time), Research to meet a scholarship/fellowship requirement', 'Research for class credit, Research for pay, Research to meet a scholarship/fellowship requirement', 'Research for pay, Research for neither credit nor pay (volunteering time)', 'Research for pay, Research for neither credit nor pay (volunteering time), Research to meet a scholarship/fellowship requirement', 'Research for class credit, Research to meet a scholarship/fellowship requirement'],
        #'Unpaid only': ['Research for class credit, Research for neither credit nor pay (volunteering time)', 'Research for neither credit nor pay (volunteering time)', ],
        #'Paid only': ['Research for pay', 'Research for pay, Research to meet a scholarship/fellowship requirement']},
    #53: {"Only selected White/Caucasian": ["White/Caucasian"], "Selected other group(s)": ['Asian', 'Black or African American', 'Black or African American, Hispanic', 'Hispanic', 'Pacific Islander, Asian', 'Prefer not to disclose', 'White/Caucasian, Asian', 'White/Caucasian, Asian, Native American', 'White/Caucasian, Hispanic', 'White/Caucasian, Native American']}
    #28: {'5-6 hours': '5-6 hours', '7-8 hours': '7-8 hours', '9-10 hours': '9-10 hours', '11-12 hours': '11-12 hours', '13-15 hours': '13-15 hours', '16-20 hours': '16-20 hours', '21-25 hours': '21-25 hours'},
    25: {'5-6 hours': '5-6 hours', '7-8 hours': '7-8 hours', '9-10 hours': '9-10 hours', '11-12 hours': '11-12 hours', '13-15 hours': '13-15 hours', '16-20 hours': '16-20 hours', '21-25 hours': '21-25 hours'},
    26: {'5-6 hours': '5-6 hours', '7-8 hours': '7-8 hours', '9-10 hours': '9-10 hours', '11-12 hours': '11-12 hours', '13-15 hours': '13-15 hours', '16-20 hours': '16-20 hours', '21-25 hours': '21-25 hours'},
    #29: {'5-6 hours': '5-6 hours', '7-8 hours': '7-8 hours', '9-10 hours': '9-10 hours'},
    37: {"0 hours": "0 hours","1-5 hours": "1-5 hours", "6-10 hours": "6-10 hours", "11-15 hours": "11-15 hours","16-20 hours": "16-20 hours","21-30 hours": "21-30 hours","31-40 hours": "31-40 hours","41-50 hours": "41-50 hours","61+ hours": "61+ hours"},
    selections[dataset]['gender']: {'Male': ['Male'], 'Female and Nonbinary': ['Female', 'Nonbinary / Third Gender', 'Man', 'Woman', ]},
    selections[dataset]['race']: {'Only selected White/Caucasian': ['White/Caucasian'], 'Other': ['Hispanic, White (latino)', 'Pacific Islander, Asian', 'Black or African American, Hispanic', 'White/Caucasian, Asian, Native American', 'White/Caucasian, Hispanic, Jewish', 'White/Caucasian, Asian', 'Asian', 'Hispanic', 'White/Caucasian, Hispanic', 'Black or African American', 'White/Caucasian, Native American', 'South Asian', ]},
    selections[dataset]['lgbtq']: {'Only selected Heterosexual': ['Heterosexual/Straight', 'Heterosexual/Straight, never think about this question'], 'Other': ['Questioning', 'Bisexual/Pansexual, Gay', 'Lesbian', 'Non-identifying', 'Asexual', 'Bisexual/Pansexual, Demisexual', 'Bisexual/Pansexual, Queer', 'Heterosexual/Straight, Asexual, Questioning', 'Bisexual/Pansexual', 'Gay', 'Straight ', 'Queer', 'Asexual, Prefer not to disclose', 'Bisexual/Pansexual, Heterosexual/Straight',]},
    selections[dataset]["us"]: {"US Education": ["Yes"], "Non-US Education": ["No"],},
    #selections[dataset]["year"]: {"2023": ["2023"], "2024": ["2024"], "2025": ["2025"], "2026": ["2026"]},
    }
#for x in [20, 21] + list(range(23,36)):
#    simplified_bins[x] = {"Strongly disagree": [1], "Disagree": [2], "Neither agree nor disagree": [3], "Agree": [4], "Strongly agree": [5]}
#for x in range(38,52):
#    simplified_bins[x] = {"Never": [1], "Rarely": [2], "Occasionally": [3], "Sometimes": [4], "Most of the time": [5]}

def isNumeric(n_q, bin_vals):
    numeric_vals = [1,2,3,4,5]
    for n in numeric_vals:
        if n in bin_vals:
            if not "many" in questions[years[0]][n_q]:
                simplified_bins[n_q] = {"Strongly disagree": [1], "Disagree": [2], "Neither agree nor disagree": [3], "Agree": [4], "Strongly agree": [5]}
            return True
    numeric_vals = ["1.0", "2.0", "3.0", "4.0", "5.0"]
    for n in numeric_vals:
        if n in bin_vals:
            if not "many" in questions[years[0]][n_q]:
                simplified_bins[n_q] = {"Strongly disagree": [1], "Disagree": [2], "Neither agree nor disagree": [3], "Agree": [4], "Strongly agree": [5]}
            return True
    return False

# Get bins for this hist
def getBins(resps, n_qs, simplified=True):

    n_q = n_qs[years[0]]
    vals = []
    for year in resps:
        vals.extend(resps[year].iloc[:, n_qs[year]])
    vals = np.array(vals)

    #print("TH: getBins()")
    #print(vals)

    #vals = np.array(resps.iloc[:, n_q])

    # Avoid issue with null responses
    cleaned_vals = vals[~pd.isnull(vals)]
    cleaned_vals = np.delete(cleaned_vals, np.where(cleaned_vals == 'nan'))
    unique_vals = np.unique(cleaned_vals)
    has_nulls = False
    if len(vals) != len(cleaned_vals):
        unique_vals = np.append(unique_vals, "No Response")
        has_nulls = True

    # If vals are numeric, use the full range
    numeric_vals = [1,2,3,4,5]
    if has_nulls: numeric_vals = ["1.0", "2.0", "3.0", "4.0", "5.0"]
    is_numeric = isNumeric(n_q, unique_vals)
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

# Get a title and split it when appropriate
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

            try:
                i = np.where(bins == comp_val)[0][0]
                bin_vals[i] += 1
                n_tot += 1
            except:
                print("No entry added in question", n_q, "for value", comp_val)
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

    resps = resps_array[years[0]]
    n_qs = {2023: n_q}
    if len(resps_array)>1:
        for year in resps_array:
            if year == 2023: continue
            n_qs[year] = getNQ(resps, resps_array[year], n_q)
            if n_qs[year]==-1: continue

    #bins = getBins(resps, n_q, simplified=simplified)
    bins = getBins(resps_array, n_qs, simplified=simplified)
    data = {}
    for sel in sels:
        data[sel] = {}
        vals = np.array(resps.iloc[:, n_q])
        data[sel]["bin_vals"], data[sel]["bin_errs"] = getHistAndErr(resps, vals, n_q, bins, sels[sel], normalize, simplified=simplified)

    datas = {2023: data}
    if len(resps_array)>1:
        for year in resps_array:
            if year == 2023: continue
            n_qs[year] = getNQ(resps, resps_array[year], n_q)
            if n_qs[year]==-1: continue
            datas[year] = {}
            for sel in sels:
                datas[year][sel] = {}
                vals = np.array(resps_array[year].iloc[:, n_qs[year]])
                datas[year][sel]["bin_vals"], datas[year][sel]["bin_errs"] = getHistAndErr(resps, vals, n_q, bins, sels[sel], normalize, simplified=simplified)

    '''
    if resps2 is not None:
        n_q2 = getNQ(resps, resps2, n_q)
        data2 = {}
        for sel in sels:
            data2[sel] = {}
            vals2 = np.array(resps2.iloc[:, n_q2])
            data2[sel]["bin_vals"], data2[sel]["bin_errs"] = getHistAndErr(resps, vals2, n_q, bins, sels[sel], normalize, simplified=simplified)
    '''

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
            patches, texts, autotexts = axs.pie(data[year][sel]["bin_vals"], labels=getBinLabels(bins, 2), autopct='%1.1f%%', textprops={'fontsize': 12}, colors=colors)
            axs.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
            axs.margins(0.5)
            for autotext in autotexts:
                autotext.set_color('white')
            axs.set_title(getSplitString(questions[years[0]][n_q]), fontsize=10)
            #plt.tight_layout()
            plt.savefig(output_dir+"q_%d%s_pie.png"%(n_q, app))
        return

    for sel in sels:
        for year in resps_array:
            mylabel = sel
            if showMean(bins, n_q):
                mylabel = "%s (mean: %s)"%(sel, getMean(datas[year][sel]["bin_vals"], nprofs=("How many faculty members" in questions[year][n_q])))
                if len(resps_array)>1: mylabel = mylabel.replace(sel, str(year))
            elif len(resps_array)>1: mylabel = str(year)
            axs.errorbar(range(len(bins)), datas[year][sel]["bin_vals"], yerr=datas[year][sel]["bin_errs"], label=mylabel, marker="o")

    '''
    for sel in sels:
        mylabel = sel
        if showMean(bins, n_q):
            mylabel = "%s (mean: %s)"%(sel, getMean(data[sel]["bin_vals"], nprofs=("How many faculty members" in questions[year][n_q])))
            if resps2 is not None: mylabel = mylabel.replace(sel, "2022")
        elif resps2 is not None: mylabel = "2022"
        axs.errorbar(range(len(bins)), data[sel]["bin_vals"], yerr=data[sel]["bin_errs"], label=mylabel, marker="o")

        if resps2 is not None:
            mylabel = mylabel.replace("2022", "2021")
            if showMean(bins, n_q):
                mylabel = "%s (mean: %s)"%(2021, getMean(data2[sel]["bin_vals"], nprofs=("How many faculty members" in questions[year][n_q])))
            axs.errorbar(range(len(bins)), data2[sel]["bin_vals"], yerr=data2[sel]["bin_errs"], label=mylabel, marker="o")
    '''

    plt.xticks(range(len(bins)), getBinLabels(bins))
    axs.margins(0.2)
    plt.ylim(bottom=0)
    if normalize: plt.ylabel("Fraction in Group")
    else: plt.ylabel("Number of Respondents")
    #if n_q in [20, 21] + list(range(23,36)) + list(range(38,52)):
    #    for i, sel in enumerate(sels):
    #        plt.text(0.02, 0.9-0.1*i, "%s: %.2f"%(sel, getMean(data[year][sel]["bin_vals"])), transform=axs.transAxes)
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
    axs.set_title(getSplitString(questions[years[0]][n_q]), fontsize=10)
    #if len(sels)>1 or resps2 is not None: plt.legend(loc='best', numpoints=1, framealpha=1) #, bbox_to_anchor=(0.5, 1.5))
    if len(sels)>1 or len(resps_array)>1: plt.legend(loc='best', numpoints=1, framealpha=1) #, bbox_to_anchor=(0.5, 1.5))
    else:
        if showMean(bins, n_q): plt.text(0.6, 0.85, "mean: %s"%(getMean(data[year][sel]["bin_vals"], ("How many faculty members" in questions[year][n_q]))), transform = axs.transAxes)
    plt.savefig(output_dir+"q_%d%s.png"%(n_q, app))
    return

def showMean(bins, n_q):
    if isNumeric(n_q, bins):
        return True
    if "Disagree" in bins:
        return True
    #if n_q in [16, 17, 18, 19, 20, 21] + list(range(23,36)) + list(range(38,52)): return True
    return False

def getAllSelections(respsonses, n_q, simplified=True):
    selections = {}
    if simplified and n_q in simplified_bins:
        for k in simplified_bins[n_q]:
            #selections[k] = "vals[j] in simplified_bins[%d]['%s']"%(n_q, k)
            selections[k] = "resps.iloc[:, %d][j] in simplified_bins[%d]['%s']"%(n_q, n_q, k)
        return selections

    values = getBins(respsonses, n_q)
    for v in values:
        if type(v)==np.int64: # Value is an int
            #selections[v] = "vals[j] == %s"%(n_q, v)
            selections[v] = "resps.iloc[:, %d][j] == %s"%(n_q, v)
        else:
            #selections[v] = "vals[j] == '%s'"%(n_q, v)
            selections[v] = "resps.iloc[:, %d][j] == '%s'"%(n_q, v)
    return selections

def plotSelections(resps, x, key):
    plotData(resps, x, getAllSelections(resps, selections[dataset][key]),
             app="_"+key)

def getNQ(resp, resp2, n_q):
    quest = resp.columns.values
    quest2 = resp2.columns.values
    key_q = quest[n_q]
    return np.where(quest2 == key_q)[0][0]

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
for year in years:

    print(f"Working with {population}s from {year}.")
    with open(resp_files[population][year], newline='') as csvfile:
        resps = pd.read_csv(csvfile)
        quests = resps.columns.values
    responses[year] = resps
    questions[year] = quests

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


#plotData(responses, 16, getAllSelections(responses, 54), app="_genders")
#plotData(responses, 16, getAllSelections(responses, 58), app="_nontrad")
#plotData(responses, 9, getAllSelections(responses, 2), app="_transfer")
#plotData(responses, 16, getAllSelections(responses, 53), app="_race")
#plotData(responses, 59, {"all":"True"}, app="_all", normalize=False, simplified=True)
#exit()


for year in years:

    #myrange = [33]
    myrange = range(len(questions[year]))
    for x in myrange:

        # Just make a simple plot of everything, no weights
        #plotData(responses[year], x, {"all":"True"}, app="_all", normalize=False, simplified=True)
        # Pie chart version
        #plotData(responses[year], x, {"all":"True"}, app="_allspec", normalize=False, pie=True, simplified=True)

        # Do comparisons
        #try: plotData(responses, x, {"all":"True"}, app="_comp", normalize=True, simplified=True)
        #except: pass
        #plotData(responses, x, {"all":"True"}, app="_comp", normalize=True, simplified=True)

        plotSelections(responses, x, "gender")
        #plotSelections(responses, x, "race")
        #plotSelections(responses, x, "lgbtq")
        #plotSelections(responses, x, "us")
        #plotSelections(responses, x, "year")

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


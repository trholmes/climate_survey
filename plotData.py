import matplotlib
matplotlib.use('Agg')
import math
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors
from matplotlib.ticker import PercentFormatter

### HELPER FUNCTIONS ###

    # Useful question numbers
    # 53 = race/ethnicity, 54 = gender, 58 = non-traditional students
    # 2 = transfers, 3 = year

# Set up simplified bins
simplified_bins = {
        8: {"Wants to go to physics grad school": ["Graduate school in physics or astronomy"], "Other": ['Considering multiple of the above options', 'Government Contractor - Threat Radar Engineering', 'Graduate school in an engineering discipline or computer science', 'Graduate school in another STEM field', 'Military', 'Not sure', 'Other graduate or professional school (medical, law, pharmacy, etc)', 'Other industry, not including Tech (e.g. optics, oil, etc.)', 'Teaching K-12',]},
    56: {"Heterosexual/Straight": ['Heterosexual/Straight'], "All Others": ['Asexual', 'Asexual, Prefer not to disclose', 'Bisexual/Pansexual', 'Bisexual/Pansexual, Demisexual', 'Bisexual/Pansexual, Queer', 'Prefer not to disclose', 'Queer', 'Questioning']},
    54: {"Man": ["Man"], "Woman & Other": ["Woman", "Nonbinary / Third Gender", "Prefer not to disclose"]},
    58: {"Non-Traditional": ["Yes"], "Traditional": ["No"]},
    11: {'Did not participate in research':['Did not participate in research'],
        'Combination paid/unpaid':['Research for class credit, Research for pay', 'Research for class credit, Research for pay, Research for a combination of credit and pay', 'Research for class credit, Research for pay, Research for neither credit nor pay (volunteering time)', 'Research for class credit, Research for pay, Research for neither credit nor pay (volunteering time), Research to meet a scholarship/fellowship requirement', 'Research for class credit, Research for pay, Research to meet a scholarship/fellowship requirement', 'Research for pay, Research for neither credit nor pay (volunteering time)', 'Research for pay, Research for neither credit nor pay (volunteering time), Research to meet a scholarship/fellowship requirement', 'Research for class credit, Research to meet a scholarship/fellowship requirement'],
        'Unpaid only': ['Research for class credit, Research for neither credit nor pay (volunteering time)', 'Research for neither credit nor pay (volunteering time)', ],
        'Paid only': ['Research for pay', 'Research for pay, Research to meet a scholarship/fellowship requirement']},
    53: {"Only selected White/Caucasian": ["White/Caucasian"], "Selected other group(s)": ['Asian', 'Black or African American', 'Black or African American, Hispanic', 'Hispanic', 'Pacific Islander, Asian', 'Prefer not to disclose', 'White/Caucasian, Asian', 'White/Caucasian, Asian, Native American', 'White/Caucasian, Hispanic', 'White/Caucasian, Native American']}
}
for x in [20, 21] + list(range(23,36)):
    simplified_bins[x] = {"Strongly disagree": [1], "Disagree": [2], "Neither agree nor disagree": [3], "Agree": [4], "Strongly agree": [5]}
for x in range(38,52):
    simplified_bins[x] = {"Never": [1], "Rarely": [2], "Occasionally": [3], "Sometimes": [4], "Most of the time": [5]}

# Get bins for this hist
def getBins(resps, n_q, simplified=True):

    vals = np.array(responses[questions[n_q]])

    # Avoid issue with null responses
    cleaned_vals = vals[~pd.isnull(vals)]
    unique_vals = np.unique(cleaned_vals)
    if len(vals) != len(cleaned_vals):
        unique_vals = np.append(unique_vals, "No Response")

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
def getHistAndErr(resps, n_q, bins, sel="True", normalize=True, simplified=True):

    vals = np.array(responses[questions[n_q]])

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

            i = np.where(bins == comp_val)[0][0]
            bin_vals[i] += 1
            n_tot += 1
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
    for i, v in enumerate(vals):
        count += v
        total += (val_index[i])*v
    if count>0: return total/count
    return -1

# Make a plot comparing response values for different selections
def plotData(resps, n_q, sels={"all":"True"}, app="", normalize=True, pie=False, simplified=True):

    bins = getBins(resps, n_q, simplified=simplified)
    data = {}
    for sel in sels:
        data[sel] = {}
        data[sel]["bin_vals"], data[sel]["bin_errs"] = getHistAndErr(resps, n_q, bins, sels[sel], normalize, simplified=simplified)

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
            #axs.pie(data[sel]["bin_vals"], explode=explode, labels=getBinLabels(bins), autopct='%1.1f%%', shadow=True, startangle=90)
            patches, texts, autotexts = axs.pie(data[sel]["bin_vals"], labels=getBinLabels(bins, 2), autopct='%1.1f%%', textprops={'fontsize': 12})
            axs.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
            axs.margins(0.5)
            for autotext in autotexts:
                autotext.set_color('white')
            axs.set_title(getSplitString(questions[n_q]), fontsize=10)
            #plt.tight_layout()
            plt.savefig("plots/q_%d%s_pie.png"%(n_q, app))
        return

    for sel in sels:
        if showMean(n_q):
            axs.errorbar(range(len(bins)), data[sel]["bin_vals"], yerr=data[sel]["bin_errs"], label="%s (mean: %.2f)"%(sel, getMean(data[sel]["bin_vals"], nprofs=(n_q in [16, 17, 18, 19]))), marker="o")
        else: axs.errorbar(range(len(bins)), data[sel]["bin_vals"], yerr=data[sel]["bin_errs"], label=sel, marker="o")
    plt.xticks(range(len(bins)), getBinLabels(bins))
    axs.margins(0.2)
    plt.ylim(bottom=0)
    if normalize: plt.ylabel("Fraction in Group")
    else: plt.ylabel("Number of Respondents")
    #if n_q in [20, 21] + list(range(23,36)) + list(range(38,52)):
    #    for i, sel in enumerate(sels):
    #        plt.text(0.02, 0.9-0.1*i, "%s: %.2f"%(sel, getMean(data[sel]["bin_vals"])), transform=axs.transAxes)
    if len(questions[n_q].split())>30:
        plt.subplots_adjust(top=(1-len(questions[n_q].split())*0.005))
    if False: # maxxlabel > 15 and maxxlabel <= 30:
        plt.xticks(rotation=10)
        axs.set_xticklabels(getBinLabels(bins), ha='right')
        plt.subplots_adjust(bottom=maxxlabel*0.007)
    if maxxlabel > 15:
        plt.xticks(rotation=60)
        axs.set_xticklabels(getBinLabels(bins), ha='right')
        plt.subplots_adjust(bottom=min(0.4,maxxlabel*0.03))
    axs.set_title(getSplitString(questions[n_q]), fontsize=10)
    if len(sels)>1: plt.legend(loc='best', numpoints=1, framealpha=1) #, bbox_to_anchor=(0.5, 1.5))
    else:
        if showMean(n_q): plt.text(0.75, 0.85, "mean: %.2f"%(getMean(data[sel]["bin_vals"], nprofs=(n_q in [16, 17, 18, 19]))), transform = axs.transAxes)
    plt.savefig("plots/q_%d%s.png"%(n_q, app))

    return

def showMean(n_q):
    if n_q in [16, 17, 18, 19, 20, 21] + list(range(23,36)) + list(range(38,52)): return True
    return False

def getAllSelections(respsonses, n_q, simplified=True):
    selections = {}
    if simplified and n_q in simplified_bins:
        for k in simplified_bins[n_q]:
            selections[k] = "responses[questions[%d]][j] in simplified_bins[%d]['%s']"%(n_q, n_q, k)
        return selections

    values = getBins(respsonses, n_q)
    for v in values:
        if type(v)==np.int64: # Value is an int
            selections[v] = "responses[questions[%d]][j] == %s"%(n_q, v)
        else:
            selections[v] = "responses[questions[%d]][j] == '%s'"%(n_q, v)
    return selections

#resp_file = "data/Physics_Undergrad_Survey (Responses) - Form Responses 1.csv"
resp_file = "data/cleaned.csv"
questions = ['Timestamp', 'Which of these describes you?', 'Did you start your college career at UT or did you transfer from elsewhere?', 'What year do you expect to graduate from UT?', 'Do you have a major or minor in another department?', 'If you answered yes to the previous question, what is your non-physics major/minor(s)?', 'How did you choose to study physics? (example: liked high school physics, physicists in the family, curiosity about the universe, anything and everything else)', 'How would you describe your participation in student organizations in the physics department?', 'What are are your plans after completing your current degree?', 'Did you know that graduate programs in physics and astronomy are typically free, and in fact pay you? (i.e. tuition is covered and students receive a stipend for research or teaching)', 'Have you participated in physics research as an undergraduate?', 'If you participated for research, which of these apply:', 'Which of these statements most applies to you?', 'Which of these funding sources have supported your research? (If none, leave blank.)', 'Have you done a professional internship that is distinct from research? ', 'Which of these activities have you participated in? (If none, leave blank.)', 'How many faculty members do you know well enough to ask them to write a letter of recommendation?', 'How many faculty members do you know well enough to ask them for academic advice?', 'How many faculty members do you know well enough to ask them for personal advice?', 'How many faculty members do you consider role models?', 'To what extent do you agree or disagree with the following statement: "I received sufficient mentoring in the physics department." (1: Strongly disagree 2: Disagree 3: Neither agree nor disagree 4: Agree 5: Strongly agree)', 'To what extent do you agree or disagree with the following statement: "My experience in the physics department adequately prepared me for my next steps after graduation." (1: Strongly disagree 2: Disagree 3: Neither agree nor disagree 4: Agree 5: Strongly agree)', 'Any other comments on the topic of research or career preparation?', 'I feel a sense of community with my classmates in the physics department. ', 'I feel a sense of community with the undergraduate physics majors as a whole. ', 'Physics instructors encourage my participation in class.', "I feel comfortable approaching my physics instructors when I don't understand a concept.", 'My physics TAs encourage my participation in class.', "I feel comfortable approaching my physics TAs when I don't understand a concept.", 'I feel supported and encouraged by physics faculty in the research I have done. (Skip if N/A)', 'I feel supported and encouraged by physics graduate students/postdocs in the research I have done. (Skip if N/A)', 'The physics department creates a supportive environment.', 'I feel like a valued member of the physics department.', 'I think about changing my major/minor away from physics.', 'I think about transferring to another institution.', 'I plan to complete my physics degree at UT.', 'If you have thought about leaving the department or university, can you describe what contributed to the desire to leave? (e.g. class load, climate, etc.)', 'Any other comments on the topic of department climate?', 'In the physics department, I personally have been treated negatively because of my race.', 'In the physics department, I personally have been treated negatively because of my gender.', 'In the physics department, I personally have been treated negatively because of my national origin.', 'In the physics department, I personally have been treated negatively because of my sexual identity.', 'In the physics department, I personally have been treated negatively because of a disability.', 'In the physics department, I have seen others treated negatively because of their race.', 'In the physics department, I have seen others treated negatively because of their gender.', 'In the physics department, I have seen others treated negatively because of their national origin.', 'In the physics department, I have seen others treated negatively because of their sexual orientation.', 'In the physics department, I have seen others treated negatively because of a disability.', 'I witness microaggressions in the physics department (to myself or others).', 'When microaggressions occur, they are acknowledged and addressed. (Leave blank if N/A)', 'I witness harassment in the physics department (to myself or others).', 'When harassment occurs, it is acknowledged and addressed. (Leave blank if N/A)', 'Any other comments about inclusion, harassment, or equity in the physics department?', 'Which groups do you belong to? (check all that apply)', 'What is your gender identity?', 'Do you identify as transgender?', 'Which best describes your sexual orientation/identity? (Select all that apply.)', 'Did you complete the majority of your K-12 education in the United States?', 'Are you a "non-traditional student" (i.e. did you have a gap of more than a year in your education career)?', 'Did any of your parents or grandparents graduate from college?', "Are there any questions you wish we'd asked? (And what would your answer to those questions be?)", "Anything else you'd like us to know?"]

# Quickly get list of enumerated questions
#for i, q in enumerate(questions):
#    print(i, q)

with open(resp_file, newline='') as csvfile:
    responses = pd.read_csv(csvfile)

    #print(getBins(responses,8))
    #exit()

    # Useful question numbers
    # 53 = race/ethnicity, 54 = gender, 58 = non-traditional students
    # 2 = transfers, 3 = year

    #plotData(responses, 16, getAllSelections(responses, 54), app="_genders")
    #plotData(responses, 16, getAllSelections(responses, 58), app="_nontrad")
    #plotData(responses, 9, getAllSelections(responses, 2), app="_transfer")
    #plotData(responses, 16, getAllSelections(responses, 53), app="_race")

    for x in range(1, 62):

        # Just make a simple plot of everything, no weights
        #plotData(responses, x, {"all":"True"}, app="_all", normalize=False)
        #plotData(responses, x, {"all":"True"}, app="_allspec", normalize=False, pie=True, simplified=False)

        #plotData(responses, x, getAllSelections(responses, 56), app="_lgbtq")
        #plotData(responses, x, getAllSelections(responses, 3), app="_years")
        #plotData(responses, x, getAllSelections(responses, 2), app="_transfers")
        #plotData(responses, x, getAllSelections(responses, 54), app="_genders")
        plotData(responses, x, getAllSelections(responses, 8), app="_career")
        #plotData(responses, x, getAllSelections(responses, 58), app="_nontrad")
        #plotData(responses, x, getAllSelections(responses, 53), app="_race")
        continue


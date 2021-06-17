import math
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors
from matplotlib.ticker import PercentFormatter

### HELPER FUNCTIONS ###

# Get bins for this hist
def getBins(resps, n_q):
    vals = np.array(responses[questions[n_q]])

    # Avoid issue with null responses
    cleaned_vals = vals[~pd.isnull(vals)]
    #cleaned_vals = np.array(filter(lambda v: v==v, vals))
    unique_vals = np.unique(cleaned_vals)
    if len(vals) != len(cleaned_vals):
        unique_vals = np.append(unique_vals, "No Response")
    return unique_vals

# Simple poisson error
def getErr(arr):
    err_arr = []
    for entry in arr:
        err_arr.append(math.sqrt(entry))
    return err_arr

# Get a title and split it when appropriate
def getSplitString(s, max_words = 10):

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
def getHistAndErr(resps, n_q, bins, sel="True", normalize="True"):

    vals = np.array(responses[questions[n_q]])

    n_tot = 0
    bin_vals = [0]*len(bins)
    for j, val in enumerate(vals):
        if eval(sel):
            comp_val = val
            if pd.isnull(val):
                comp_val = "No Response"
            elif "No Response" in bins:
                comp_val = str(val)
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


# Make a plot comparing response values for different selections
def plotData(resps, n_q, sels=["True"], app="", normalize=True):

    bins = getBins(resps, n_q)
    data = {}
    for sel in sels:
        data[sel] = {}
        data[sel]["bin_vals"], data[sel]["bin_errs"] = getHistAndErr(resps, n_q, bins, sels[sel], normalize)

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
    for sel in sels:
        axs.errorbar(range(len(bins)), data[sel]["bin_vals"], yerr=data[sel]["bin_errs"], label=sel, marker="o")
    plt.xticks(range(len(bins)), getBinLabels(bins))
    axs.margins(0.2)
    if len(questions[n_q].split())>30:
        plt.subplots_adjust(top=(1-len(questions[n_q].split())*0.005))
    if False: # maxxlabel > 15 and maxxlabel <= 30:
        plt.xticks(rotation=10)
        axs.set_xticklabels(getBinLabels(bins), ha='right')
        plt.subplots_adjust(bottom=maxxlabel*0.007)
    if maxxlabel > 15:
        plt.xticks(rotation=60)
        axs.set_xticklabels(getBinLabels(bins), ha='right')
        plt.subplots_adjust(bottom=min(0.5,maxxlabel*0.05))
    axs.set_title(getSplitString(questions[n_q]), fontsize=10)
    plt.legend(loc='best', numpoints=1, framealpha=1) #, bbox_to_anchor=(0.5, 1.5))
    plt.savefig("plots/q_%d%s.png"%(n_q, app))

    return

def getAllSelections(respsonses, n_q):
    values = getBins(respsonses, n_q)
    selections = {}
    for v in values:
        if type(v)==np.int64: # Value is an int
            selections[v] = "responses[questions[%d]][j] == %s"%(n_q, v)
        else:
            selections[v] = "responses[questions[%d]][j] == '%s'"%(n_q, v)
    return selections

resp_file = "data/Physics_Undergrad_Survey (Responses) - Form Responses 1.csv"
questions = ['Timestamp', 'Which of these describes you?', 'Did you start your college career at UT or did you transfer from elsewhere?', 'What year do you expect to graduate from UT?', 'Do you have a major or minor in another department?', 'If you answered yes to the previous question, what is your non-physics major/minor(s)?', 'How did you choose to study physics? (example: liked high school physics, physicists in the family, curiosity about the universe, anything and everything else)', 'How would you describe your participation in student organizations in the physics department?', 'What are are your plans after completing your current degree?', 'Did you know that graduate programs in physics and astronomy are typically free, and in fact pay you? (i.e. tuition is covered and students receive a stipend for research or teaching)', 'Have you participated in physics research as an undergraduate?', 'If you participated for research, which of these apply:', 'Which of these statements most applies to you?', 'Which of these funding sources have supported your research? (If none, leave blank.)', 'Have you done a professional internship that is distinct from research? ', 'Which of these activities have you participated in? (If none, leave blank.)', 'How many faculty members do you know well enough to ask them to write a letter of recommendation?', 'How many faculty members do you know well enough to ask them for academic advice?', 'How many faculty members do you know well enough to ask them for personal advice?', 'How many faculty members do you consider role models?', 'To what extent do you agree or disagree with the following statement: "I received sufficient mentoring in the physics department." (1: Strongly disagree 2: Disagree 3: Neither agree nor disagree 4: Agree 5: Strongly agree)', 'To what extent do you agree or disagree with the following statement: "My experience in the physics department adequately prepared me for my next steps after graduation." (1: Strongly disagree 2: Disagree 3: Neither agree nor disagree 4: Agree 5: Strongly agree)', 'Any other comments on the topic of research or career preparation?', 'I feel a sense of community with my classmates in the physics department. ', 'I feel a sense of community with the undergraduate physics majors as a whole. ', 'Physics instructors encourage my participation in class.', "I feel comfortable approaching my physics instructors when I don't understand a concept.", 'My physics TAs encourage my participation in class.', "I feel comfortable approaching my physics TAs when I don't understand a concept.", 'I feel supported and encouraged by physics faculty in the research I have done. (Skip if N/A)', 'I feel supported and encouraged by physics graduate students/postdocs in the research I have done. (Skip if N/A)', 'The physics department creates a supportive environment.', 'I feel like a valued member of the physics department.', 'I think about changing my major/minor away from physics.', 'I think about transferring to another institution.', 'I plan to complete my physics degree at UT.', 'If you have thought about leaving the department or university, can you describe what contributed to the desire to leave? (e.g. class load, climate, etc.)', 'Any other comments on the topic of department climate?', 'In the physics department, I personally have been treated negatively because of my race.', 'In the physics department, I personally have been treated negatively because of my gender.', 'In the physics department, I personally have been treated negatively because of my national origin.', 'In the physics department, I personally have been treated negatively because of my sexual identity.', 'In the physics department, I personally have been treated negatively because of a disability.', 'In the physics department, I have seen others treated negatively because of their race.', 'In the physics department, I have seen others treated negatively because of their gender.', 'In the physics department, I have seen others treated negatively because of their national origin.', 'In the physics department, I have seen others treated negatively because of their sexual orientation.', 'In the physics department, I have seen others treated negatively because of a disability.', 'I witness microaggressions in the physics department (to myself or others).', 'When microaggressions occur, they are acknowledged and addressed. (Leave blank if N/A)', 'I witness harassment in the physics department (to myself or others).', 'When harassment occurs, it is acknowledged and addressed. (Leave blank if N/A)', 'Any other comments about inclusion, harassment, or equity in the physics department?', 'Which groups do you belong to? (check all that apply)', 'What is your gender identity?', 'Do you identify as transgender?', 'Which best describes your sexual orientation/identity? (Select all that apply.)', 'Did you complete the majority of your K-12 education in the United States?', 'Are you a "non-traditional student" (i.e. did you have a gap of more than a year in your education career)?', 'Did any of your parents or grandparents graduate from college?', "Are there any questions you wish we'd asked? (And what would your answer to those questions be?)", "Anything else you'd like us to know?"]

with open(resp_file, newline='') as csvfile:
    responses = pd.read_csv(csvfile)

    # Useful question numbers
    # 53 = race/ethnicity, 54 = gender, 58 = non-traditional students
    # 2 = transfers, 3 = year

    #plotData(responses, 16, getAllSelections(responses, 54), app="_genders")
    #plotData(responses, 16, getAllSelections(responses, 58), app="_nontrad")
    plotData(responses, 9, getAllSelections(responses, 2), app="_transfer")
    #plotData(responses, 16, getAllSelections(responses, 53), app="_race")

    for x in range(62):
        #plotData(responses, x, getAllSelections(responses, 3), app="_years")
            #plotData(responses, x, getAllSelections(responses, 2), app="_transfers")
        plotData(responses, x, getAllSelections(responses, 54), app="_genders")
            #plotData(responses, x, getAllSelections(responses, 58), app="_nontrad")
            #plotData(responses, x, getAllSelections(responses, 53), app="_race")
        #except:
        #    continue



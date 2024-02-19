In general, this script is run by running `plotData.py`. At the bottom of this file, you'll find several example calls to the main functions. 

When starting with a new dataset, you'll need to first populate `selections.json` with some metatdata about that dataset. In particular, any of the demographic questions that you plan to use to make comparative plots will need to be included by keyword here. Typically I include gender, race, year of entry / expected graduation year (depending if grad or undergrad), LGBTQIA+ status, and whether they completed the majority of their education in the US. 
You'll also add the file path to the CV you download from the google form here. A good way to quickly get the data you need to populate this file is by turning on the `dump_questions` option in `plotData.py` and running it.

In addition to populating these questions numbers, you also need to decide how you'll build large enough groups to be non-identifying when you plot. `getSelections.py` is an attempt at streamlining this (though it could be much better streamlined). This gathers up all the race/gender/LGBTQIA+ responses and puts them into simpler groups, essentially are you part of the ``dominant'' physics group (white, male, cis-het). If we ever had enough stats it would be great to separate further,
but for now I have just been using two categories in these areas. You can take the output of this script and just paste it into the top of `plotData.py` where `simplified_bins` is being defined. This is sloppy and could obviously be done in a more automated way. To run using these simplified bins rather than plotting every category separately, you set `simplified=True` when calling a function in `plotData.py`. 

You may also want to update the years going into the simplified bins to
represent the full space. It also is different for grads and undergrads. The reason they're in here explicitly was to make sure they showed up in this order, and without gaps. You could also comment the line out entirely to just let it go by the data. 

Once all of this is set up for a dataset, the main functions you'll want to use are at the bottom of `plotData.py`. To plot one year only, set the `years` variable to a list with just that value. To plot multiple years in comparison, you can put them all in that list, but don't try to do other demographic group comparisons at the same time. You can find some examples at the very bottom of the script of how you might track a given population's experiences over time by making use
of the selections option. This is a little bit un-tested, so please verify that the results are right if using it.



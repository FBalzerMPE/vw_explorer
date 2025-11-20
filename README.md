# VIRUS-W Explorer

Simple tool to review VIRUS-W observations on-the-fly at the Harlan-J-Smith telescope at McDonald Observatory.

## Installation and requirements

Since the linux machine operating VIRUS-W runs on a specific python-3.6 version, the script should be backwards-compatible but may also work with newer python versions.

You can install it by cloning into this repository and then modifying 

## Functionalities

### Observation log parsing

This tool allows to parse an observation log file given that it's in the correct format.

A representation of these is then saved as a .csv file within the DATAPATH.

```python
import vw_guider_analysis as vwg
observations = vwg.load_observations(logfile_path="YOUR_PATH/log.txt")
# Generate a sequence object
obs_sequence = vwg.ObservationSequence(observations)
# Print a summary of the sequence
print(obs_sequence)
```

### Guider Frame analysis

If you provide a path to your guider frames (which may contain subdirectories for each days), you'll first need to create a Guider Index file (mapping the UT date to the file path). This can be done using
```python
vwg.create_guider_index(guider_dir="GuiderDirectory", silent=False)
```
and it may take a bit of time (~1 min per 500 guider frames) as the times need to be read from every fits header.

With the observations loaded in the previous step, you may now run some analysis for each observation. For this task, you can invoke the `GuiderSequence` class, which loads all `GuiderFrame`s for a given `Observation` object and tries to perform a fit for each.

You can then invoke diagnostic plots for the fits of these Guider Frames, and get the mean (sigma-clipped) position and FWHM.

You may also use
```python
vwg.process_observations(observations)
```
to produce such plots for each observation.

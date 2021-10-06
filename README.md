# ML-Based Marine Spill Detector

<span class="c3 c5">The proposal of the project is published in the NASA Space Apps challenge website. </span></p>

# Proposal
<p class="c8 c5"><span class="c3 c5"></span></p><p class="c5 c6"><span class="c4 c5">ML-Based Marine Spill Detector</span></p><p class="c5 c8"><span class="c3 c5"></span></p><p class="c20 c5"><span class="c3 c5">We propose an oil spill detection web application that periodically reads NASA satellite data and generates a global geographic heatmap. The purpose of the heatmap is to indicate the occurrence chance of the spillage incidents using regular deep learning methods. Our method mainly consists of combining open source datasets for the NASA Space Apps to help natural disasters decision makers plan and allocate resources efficiently. Our vision is to have the contamination accumulation in the food chain de-accelerated, and the mission of the public user interface (SaudiSpaceShuttle.com) is to provide the first responders community with an additional deep model as well as to increase the public awareness about water pollution. </span></p><p class="c9"><span class="c7">When it comes to water pollution, what goes around really comes around, since </span><span class="c7 c5">water pollution can put people at risk of diseases. </span><span class="c7">One of the most pressing sources of water pollution is the oil spill problems. Even though the challenge of monitoring sea pollution has been addressed by many organizations, the stakes are too high, and additional resources could make a difference to solve the </span><span class="c7 c5">accumulating</span><span class="c3">&nbsp;contamination in the food chain. There exist specialized monitoring organizations that use machine learning to reduce the impact of the oil spill. For instance, the National Oceanic and Atmospheric Administration, or NOAA for short, trains machine learning models to predict the trajectory of the oil spill spread. However, some spillages become irreversible within a couple of days after an oil tanker accident, and the negative effects from some types of oil could last for decades. </span></p><p class="c9"><span class="c3">Therefore, the problem statement is about providing an additional oil spill model to double check the safety of the marine resources. However, our machine learning model uses existing open data from governmental organizations. The approach we adopt to addressing the water pollution is providing heat maps that can be easily read by emergency responders and relevant decision makers. For the end user, the workflow is as friendly as entering the area of interest to the Saudi Space Shuttle web app and receiving the requested information or having it posted to the public. Nevertheless, the underlying modeling is moderately sophisticated. The everyday data acquisition, cleaning, clustering, and mapping are part of the processing method.</span></p><p class="c11 c5 c16"><span class="c7">In the ML-Based Marine Spill Detector, several pieces of open data are utilized, and two datasets are joined for extracting the main features for the marine oil-spill, chemical, and other incidents. The regular satellite imagery of the Earth API was prone to cloud blocking. Accordingly, the Synthetic Aperture Radar (SAR), from European Space Agency (ESA), which has a </span><span class="c7 c19"><a class="c1" href="https://www.google.com/url?q=https://en.wikipedia.org/wiki/European_Space_Agency%23NASA&amp;sa=D&amp;source=editors&amp;ust=1633489504725000&amp;usg=AOvVaw2w39frJhpoOh3NlFg43b8M">long-lasting partnership</a></span><span class="c7">&nbsp;with NASA, was practical since it is less variant to weather conditions, and so it has </span><span class="c7 c19"><a class="c1" href="https://www.google.com/url?q=https://sentinel.esa.int/web/sentinel/user-guides/sentinel-1-sar/applications/mapping-applications-s1-modes&amp;sa=D&amp;source=editors&amp;ust=1633489504725000&amp;usg=AOvVaw3PLl0eZmS8G2qVTZ20H-p5">multiple applications</a></span><span class="c7">, one of which is oil pollution detection. In 2023, a new SAR addition is expected to be added by NASA&#39;s joint NASA ISRO SAR Mission (</span><span class="c7 c19"><a class="c1" href="https://www.google.com/url?q=https://nisar.jpl.nasa.gov/&amp;sa=D&amp;source=editors&amp;ust=1633489504726000&amp;usg=AOvVaw2ZtMURTEw--KDhgVOLiDWd">NISAR</a></span><span class="c7">) program. In any case, the ability of reading the satellite imagery and radar readings isn&#39;t adequate without the historical data of the marine pollution incidents. For that reason, the ESA dataset was joined with the Raw Incident Data, which is open and is accessible via the IncidentNews website of National Oceanic and Atmospheric Administration (NOAA). In 2016. NASA launched an advanced weather satellite for NOAA. Nevertheless, the ESA data, along with several other datasets, is </span><span class="c7 c19"><a class="c1" href="https://www.google.com/url?q=https://developers.google.com/earth-engine/datasets&amp;sa=D&amp;source=editors&amp;ust=1633489504726000&amp;usg=AOvVaw21jbBBNePdVKVTr4VRt6uu">accessible</a></span><span class="c7">&nbsp;through Google&#39;s Earth Engine API. It needs filtering libraries, available in </span><span class="c7 c19"><a class="c1" href="https://www.google.com/url?q=https://developers.google.com/earth-engine/tutorials/community/intro-to-python-api-guiattard&amp;sa=D&amp;source=editors&amp;ust=1633489504726000&amp;usg=AOvVaw2tDeShnWTxOW1jx5sMhWHv">Python</a></span><span class="c7">&nbsp;and </span><span class="c7 c19"><a class="c1" href="https://www.google.com/url?q=https://developers.google.com/earth-engine/tutorials/tutorial_api_01&amp;sa=D&amp;source=editors&amp;ust=1633489504726000&amp;usg=AOvVaw0sHqhflDMLV8GlJTG3S7sm">JavaScript</a></span><span class="c3">, to prepare the data for downloads, and so does the Earth API. </span></p><p class="c11 c16 c5"><span class="c7">We joined the NOAA dataset to ESA data to produce the imagery data, which becomes open to the public when it is published to the website, for our deep learning model. The website is </span><span class="c7 c19"><a class="c1" href="https://www.google.com/url?q=http://www.saudispaceshuttle.com&amp;sa=D&amp;source=editors&amp;ust=1633489504727000&amp;usg=AOvVaw0dYg2FBMDPIieB6muzwV9v">www.SaudiSpaceShuttle.com</a></span><span class="c3">. Our model also trains on the pre-processed ROBORDER data from MultiMoDal Data Fusion and Analytics Group. The ROBORDER data is accessible upon requesting the research group. Please find the list of the utilized data below.</span></p><p class="c5 c11"><span class="c3">The utilized data:</span></p><ol class="c23 lst-kix_yn5vlkymikgj-0 start" start="1"><li class="c5 c12 li-bullet-0"><span class="c7">NASA, Earth API, [URL] </span><span class="c7 c19"><a class="c1" href="https://www.google.com/url?q=https://api.nasa.gov/&amp;sa=D&amp;source=editors&amp;ust=1633489504727000&amp;usg=AOvVaw0yk7uSz7p8GwRgr_-pj2zd">https://api.nasa.gov</a></span></li><li class="c12 c5 li-bullet-0"><span class="c7">ESA, Google Earth Engine API, [URL] </span><span class="c7 c19"><a class="c1" href="https://www.google.com/url?q=https://developers.google.com/earth-engine/datasets/catalog/COPERNICUS_S1_GRD&amp;sa=D&amp;source=editors&amp;ust=1633489504727000&amp;usg=AOvVaw09pc10ddOuvXya168hRQKI">https://developers.google.com/earth-engine/datasets/catalog/COPERNICUS_S1_GRD</a></span></li><li class="c12 c5 li-bullet-0"><span class="c7">NOAA, Raw Incident Data, [URL] </span><span class="c7 c19"><a class="c1" href="https://www.google.com/url?q=https://incidentnews.noaa.gov/raw/index&amp;sa=D&amp;source=editors&amp;ust=1633489504728000&amp;usg=AOvVaw3JGR6YmmWXFilxiycIYOA9">https://incidentnews.noaa.gov/raw/index</a></span></li><li class="c12 c5 li-bullet-0"><span class="c7">&quot;MultiMoDal Data Fusion and Analytics Group,&quot; ROBORDER Dataset, [1,2], [URL] </span><span class="c7 c19"><a class="c1" href="https://www.google.com/url?q=https://m4d.iti.gr/oil-spill-detection-dataset&amp;sa=D&amp;source=editors&amp;ust=1633489504728000&amp;usg=AOvVaw2TnlAZvr8s9_o-xDoW727t">https://m4d.iti.gr/oil-spill-detection-dataset</a></span><span class="c7">&nbsp;</span></li></ol><p class="c10"><span class="c7">&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Both Python and JavaScript are open-source softwares, and so are the libraries, like Earth Engine, Geopandas/Pandas, Cupy/Numpy, and Tensorflow/Keras, and the operating system as well, which is Linux. We are using Google Cloud Platform to host and accelerate the processing, since some tasks require scaling up the computational capabilities. Our time is the main asset. The Saudi Space Shuttle (SSSh) app is a non-profit project, and its minimal viable service is displaying the oil detection chances of occurrences on a heatmap over a map object, by scanning specific areas of interest that are specified by the end-users through email or a form on the website. </span></p><p class="c0"><span class="c3"></span></p><p class="c9"><span class="c7">In summary, the SSSh app gathers different types of data that can be joined geographically per the timestamp of the events, and </span><span class="c7 c5">its model is a binary classifier that trains to detect whether an incident is present or not in each of the satellite radar images using deep learning. The main tools were outlined. A Q/A table is included. </span></p><p class="c0"><span class="c3"></span></p><p class="c9"><span class="c4">Kind Regards</span></p><p class="c9"><span class="c3">The Saudi Space Shuttle Team</span></p><p class="c8" dir="rtl"><span class="c3 c5"></span></p><p class="c8" dir="rtl"><span class="c3 c5"></span></p><p class="c8" dir="rtl"><span class="c3 c5"></span></p><p class="c8" dir="rtl"><span class="c3 c5"></span></p><p class="c8" dir="rtl"><span class="c3 c5"></span></p><p class="c8"><span class="c3 c5"></span></p><p class="c8"><span class="c3 c5"></span></p><p class="c8"><span class="c3 c5"></span></p><p class="c8"><span class="c3 c5"></span></p><p class="c8"><span class="c3 c5"></span></p><p class="c8"><span class="c3 c5"></span></p><p class="c8"><span class="c3 c5"></span></p><p class="c8" dir="rtl"><span class="c3 c5"></span></p><p class="c10"><span class="c4">Table Q/A</span></p><p class="c0"><span class="c3"></span></p><a id="t.69c9aec6e1d134400c23d10e2159c61757b6dae9"></a><a id="t.0"></a><table class="c21"><tbody><tr class="c24"><td class="c18" colspan="1" rowspan="1"><p class="c0"><span class="c3"></span></p></td><td class="c2" colspan="1" rowspan="1"><p class="c10"><span class="c3">Question</span></p></td><td class="c15" colspan="1" rowspan="1"><p class="c10"><span class="c3">Answer</span></p></td></tr><tr class="c24"><td class="c18" colspan="1" rowspan="1"><p class="c10"><span class="c3">1</span></p></td><td class="c2" colspan="1" rowspan="1"><p class="c10"><span class="c3">What is the framing question of your analysis, or the purpose of the model/system you plan to build? </span></p></td><td class="c15" colspan="1" rowspan="1"><p class="c10"><span class="c3">The problem statement is about adding a new model to the existing research body. The purpose of the model is to alert emergency responders about environmental incidents as the relevant satellite readings become available. </span></p></td></tr><tr class="c24"><td class="c18" colspan="1" rowspan="1"><p class="c10"><span class="c3">2</span></p></td><td class="c2" colspan="1" rowspan="1"><p class="c10"><span class="c3">Who benefits from exploring this question or building this model/system?</span></p></td><td class="c15" colspan="1" rowspan="1"><p class="c10"><span class="c3">It can lead to prosperity for the well-being of human beings since it addresses the human food chain and marine pollution. </span></p></td></tr><tr class="c24"><td class="c18" colspan="1" rowspan="1"><p class="c10"><span class="c3">3</span></p></td><td class="c2" colspan="1" rowspan="1"><p class="c10"><span class="c3">What dataset(s) do you plan to use, and how will you obtain the data? </span></p><p class="c0"><span class="c3"></span></p></td><td class="c15" colspan="1" rowspan="1"><p class="c10"><span class="c3">The data is SAR Sentinel-1, and it is obtainable via the Google Earth Engine APIs. </span></p><p class="c0"><span class="c3"></span></p></td></tr><tr class="c24"><td class="c18" colspan="1" rowspan="1"><p class="c10"><span class="c3">4</span></p></td><td class="c2" colspan="1" rowspan="1"><p class="c10"><span class="c3">What is an individual sample/unit of analysis in this project? What characteristics/features do you expect to work with? </span></p></td><td class="c15" colspan="1" rowspan="1"><p class="c10"><span class="c3">They are Satellite images. The samples have three band features: VV, HV, and angle. </span></p><p class="c0"><span class="c3"></span></p></td></tr><tr class="c24"><td class="c18" colspan="1" rowspan="1"><p class="c10"><span class="c3">5</span></p></td><td class="c2" colspan="1" rowspan="1"><p class="c10"><span class="c3">If modeling, what will you predict as your target? </span></p></td><td class="c15" colspan="1" rowspan="1"><p class="c10"><span class="c3">The output is either Oil-Spill or No-Oil-Spill.</span></p><p class="c0"><span class="c3"></span></p></td></tr><tr class="c24"><td class="c18" colspan="1" rowspan="1"><p class="c10"><span class="c3">6</span></p></td><td class="c2" colspan="1" rowspan="1"><p class="c10"><span class="c3">How do you intend to meet the tools requirement of the project? </span></p><p class="c0"><span class="c3"></span></p></td><td class="c15" colspan="1" rowspan="1"><p class="c10"><span class="c3">They&rsquo;re mostly free.</span></p><p class="c0"><span class="c3"></span></p></td></tr><tr class="c24"><td class="c18" colspan="1" rowspan="1"><p class="c10"><span class="c3">7</span></p></td><td class="c2" colspan="1" rowspan="1"><p class="c10"><span class="c3">Are you planning in advance to need or use additional tools beyond those required? </span></p><p class="c0"><span class="c3"></span></p></td><td class="c15" colspan="1" rowspan="1"><p class="c10"><span class="c3">Yes, we might scale up the tools should it be needed. </span></p><p class="c0"><span class="c3"></span></p></td></tr><tr class="c24"><td class="c18" colspan="1" rowspan="1"><p class="c10"><span class="c3">8</span></p></td><td class="c2" colspan="1" rowspan="1"><p class="c10"><span class="c3">What would a minimum viable product (MVP) look like for this project? </span></p></td><td class="c15" colspan="1" rowspan="1"><p class="c10"><span class="c3">It is a website with a JavaScript heatmap that reads the data from the server, and the data is updated throughout the day. </span></p></td></tr></tbody></table><p class="c0"><span class="c3"></span></p><p class="c20"><span class="c5 c14">References: </span><span class="c3 c5">[1]. Krestenitis, M., Orfanidis, G., Ioannidis, K., Avgerinakis, K., Vrochidis, S., &amp; Kompatsiaris, I. (2019). Oil Spill Identification from Satellite Images Using Deep Neural Networks. Remote Sensing, 11(15), 1762. [2]. Krestenitis, M., Orfanidis, G., Ioannidis, K., Avgerinakis, K., Vrochidis, S., &amp; Kompatsiaris, I. (2019, January). Early Identification of Oil Spills in Satellite Images Using Deep CNNs. In International Conference on Multimedia Modeling (pp. 424-435). Springer, Cham.</span></p>



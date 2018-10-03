import pandas as pd
import numpy as np
import json
from flask import Flask, render_template, request
#pd.options.display.max_columns=25

app = Flask(__name__, static_url_path='/static')

#Standard home page. 'index.html' is the file in your templates that has the CSS and HTML for your app
@app.route('/', methods=['GET', 'POST'])
def index():
	return render_template('index.html')

#My home page redirects to recommender.html where the user fills out a survey (user input)
@app.route('/input_holds', methods=['GET', 'POST'])
def recommender():
	return render_template('input_holds.html')

#After they submit the survey, the recommender page redirects to recommendations.html
@app.route('/return_grade', methods=['GET', 'POST'])
def input_to_output_app():
	import pickle
	import numpy as np
	#h2o.init(ip = "localhost", port =8080)
	filename = 'RF_model_dep40.pickle'
	loaded_model = pickle.load(open(filename, 'rb'))
	df_names_holds = pd.read_csv("names_holds.csv",sep=',',header=None, names=['idx','setter','nholds','holds','Grade'])

#	filename = '../RF_model.pickle'
	#filename = 'model'
	#loaded_model = h2o.load_model(filename)
	font = ['5+','6A','6A+','6B','6B+','6C','6C+','7A','7A+','7B','7B+','7C','7C+','8A','8A+','8B','8B+']
	vgrd = [2,3,3,4,4,5,5,6,7,8,8,9,10,11,12,13,14]
	grade_conversion = dict(zip(font,vgrd))
	def alphabet_to_num(char_lst):
		num = np.array([ord(char.lower()) for char in char_lst])
		return num - 96
	def split_xy(str_arr):
		import re
		r = re.compile("([a-zA-Z]+)([0-9]+)")
		#str_arr = [x.decode('utf8').strip() for x in str_arr]
		str_arr = [x.strip() for x in str_arr]
		ret = [list(r.match(string).groups()) for string in str_arr]
		x = [ret[i][0] for i in range(len(str_arr))]
		y = [ret[i][1] for i in range(len(str_arr))]
		return np.array(x),np.array(y)
	def coord(hold_lst):
		xcoord , ycoord = split_xy(hold_lst)
		coord1 = np.vstack((alphabet_to_num(xcoord).astype(int),ycoord.astype(int)))
    # return [[x1,y1],[x2,y2]...]
		return coord1.transpose()
	def minpath_length(interm, start = None, end = None):
    #compute distance matrix
		from sklearn.metrics.pairwise import pairwise_distances
    #from skimage.graph import route_through_array
    # create numbered coordinates of interm
		interm = coord(interm)
		sort_int = np.array(sorted(interm, key=lambda x: x[1]))
		if start == None:
			start = sort_int[0]
			sort_int = np.delete(sort_int,0,0)
		elif len(start) == 1:
			start = coord(start)
		else:
			s1 = coord(start)
			start = (s1[0] + s1[1]) / 2.
		if end == None:
			end = sort_int[-1]
			sort_int = np.delete(sort_int,-1,0)
		elif len(end) == 1:
			end = coord(end)
		else:
			e1 = coord(end)
			end = (e1[0] + e1[1]) / 2.
    #print(sort_int)
		tgt = np.vstack((np.vstack((start,sort_int)),end))
		dist_matrix = pairwise_distances(tgt)
    #dist_matrix = dist_matrix[~np.eye(dist_matrix.shape[0],dtype=bool)].reshape(dist_matrix.shape[0],-1)

    #print(dist_matrix)
		n = len(sort_int) + 1
		import itertools
		permu = list(itertools.permutations(range(1,n)))
		min_dist = 1000
		min_dist_std = 100000
		for i in range(len(permu)):
			it = permu[i]
			tmp = dist_matrix[0,it[0]] + dist_matrix[it[-1],n]
			tmp_std = dist_matrix[0,it[0]]**2. + dist_matrix[it[-1],n]**2.
			for j in range(n-2):
				tmp += dist_matrix[it[j],it[j+1]]
				tmp_std += dist_matrix[it[j],it[j+1]]**2.
			min_dist = min(min_dist, tmp)
			if min_dist == tmp:
				min_dist_std = tmp_std
		min_dist_std = np.sqrt(min_dist_std / (n) - (min_dist/(n))**2. )
    #remove diagonal?
		return [min_dist, min_dist_std]

#	def alphabet_to_num(char):
#		return ord(char.lower()) - 96
#	def split_xy(string):
#		import re
#		r = re.compile("([a-zA-Z]+)([0-9]+)")
    #strings = ['foofo21', 'bar432', 'foobar12345']
#		return list(r.match(string).groups())
#	def xlims(list):
#		n = len(list)
#		coords = [alphabet_to_num(split_xy(list[i])[0]) for i in range(n)]
#		return [min(coords),max(coords)]
	def xlims(list):
    # return (leftmost, rightmost) x coordinate of a given problem
		n = len(list)
    #coords = [alphabet_to_num(split_xy(list[i])[0]) for i in range(n)]
		coords = coord(list).transpose()
#    xcoords = [coords[i][0] for i in range(n)]
#    xcoords = [alphabet_to_num(coords[i]) for i in range(n)]
		return [np.min(coords[0]),np.max(coords[0])]
# generate classes for mlb!
	mlb_class = []
	for i in range(11):
		for j in range(18):
			mlb_class.append(chr(i+65)+'%d'%(j+1))
	from sklearn.preprocessing import MultiLabelBinarizer

	mlb = MultiLabelBinarizer(classes=mlb_class)

	def sort_diff(lst):
		lst = coord(lst)
		sort_lst = np.array(sorted(lst, key=lambda x:x[1]))
		sort_diff = np.diff(sort_lst,axis=0)
		return sort_diff
	def find_user(lst,grade):
		from ast import literal_eval
		leng = len(lst)
		inp = sort_diff(lst)
		if grade == 5:
			setter_e = df_names_holds['setter'].loc[(df_names_holds['nholds']==leng) & (df_names_holds['Grade']==grade)].reset_index(drop=True)
			holds_e = df_names_holds['holds'].loc[(df_names_holds['nholds']==leng)&(df_names_holds['Grade']==grade)].reset_index(drop=True)
			grd_e = df_names_holds['Grade'].loc[(df_names_holds['nholds']==leng) & (df_names_holds['Grade']==grade)].reset_index(drop=True)
		else:
			setter_e = df_names_holds['setter'].loc[(df_names_holds['nholds']==leng) & (df_names_holds['Grade']==grade-1)].reset_index(drop=True)
			holds_e = df_names_holds['holds'].loc[(df_names_holds['nholds']==leng)&(df_names_holds['Grade']==grade-1)].reset_index(drop=True)
			grd_e = df_names_holds['Grade'].loc[(df_names_holds['nholds']==leng) & (df_names_holds['Grade']==grade-1)].reset_index(drop=True)
		if grade == 9:
			setter_h = df_names_holds['setter'].loc[(df_names_holds['nholds']==leng) & (df_names_holds['Grade']==grade)].reset_index(drop=True)
			holds_h = df_names_holds['holds'].loc[(df_names_holds['nholds']==leng)&(df_names_holds['Grade']==grade)].reset_index(drop=True)
			grd_h = df_names_holds['Grade'].loc[(df_names_holds['nholds']==leng) & (df_names_holds['Grade']==grade)].reset_index(drop=True)
		else:
			setter_h = df_names_holds['setter'].loc[(df_names_holds['nholds']==leng) & (df_names_holds['Grade']==grade+1)].reset_index(drop=True)
			holds_h = df_names_holds['holds'].loc[(df_names_holds['nholds']==leng)&(df_names_holds['Grade']==grade+1)].reset_index(drop=True)
			grd_h = df_names_holds['Grade'].loc[(df_names_holds['nholds']==leng) & (df_names_holds['Grade']==grade+1)].reset_index(drop=True)

    #print(holds_[0])
		holds_e = holds_e.apply(literal_eval)
		holds_h = holds_h.apply(literal_eval)
		comp_e = [sort_diff(holds_e[i]) for i in range(len(holds_e))]
		comp_h = [sort_diff(holds_h[i]) for i in range(len(holds_h))]

    #print(comp[1])
    #from sklearn.metrics.pairwise import cosine_similarity
		from scipy.spatial.distance import cdist
		sim_e = np.zeros((len(comp_e)))
		sim_h = np.zeros((len(comp_h)))
		for i in range(leng-1):
			for j in range(len(comp_e)):
				sim_e[j] += 1.-cdist(np.reshape(inp[i],(1,2)),np.reshape(comp_e[j][i],(1,2)),'cosine')
			for k in range(len(comp_h)):
				sim_h[k] += 1.-cdist(np.reshape(inp[i],(1,2)),np.reshape(comp_h[k][i],(1,2)),'cosine')
    #print(sim)
		index_e = np.nanargmax(sim_e)
		index_h = np.nanargmax(sim_h)
    #from sklearn.metrics.pairwise import cosine_similarity
		return [grd_e[index_e],setter_e[index_e],holds_e[index_e],grd_h[index_h],setter_h[index_h],holds_h[index_h]]


# Using Skicit-learn to split data into training and testing sets
# Instantiate model with 1000 decision trees
# Train the model on training data

	def input_to_output(lst):
		path = minpath_length(lst)
		df_path = pd.DataFrame([path], columns=list('ab'))
		print(df_path)
		h_enc = mlb.fit_transform([lst])
		h_enc = pd.DataFrame(h_enc,columns=mlb.classes_)
		tmp = xlims(lst)
		tmp = {'lr':[tmp]}
		tmp2 = pd.DataFrame.from_dict(tmp)
		tmp2[['l','r']] = pd.DataFrame(tmp2.lr.values.tolist(), index= tmp2.index)
		tmp2 = tmp2.drop(['lr'],axis=1)
		h_enc['Length'] = pd.Series(len(lst))
		h_enc = h_enc.join(tmp2)
		h_enc['width'] = h_enc['r']-h_enc['l']
		h_enc = h_enc.join(df_path)
	#	h_enc = h2o.H2OFrame(h_enc)	
		prob = loaded_model.predict(h_enc)
		#return [rf.predict(h_enc),rf.predict_proba(h_enc)]
		return prob

    # take input from recommendation.html
#	s = str(request.form['hold_coordinates'])
	lst = (request.form.getlist('check'))
	#lst = s.strip().split(',')
#	prob = input_to_output(lst)
#	pred = prob['predict'].as_data_frame().astype(int)
#	cos_sim = pred
	cos_sim = int(input_to_output(lst))
	[Grade_e,setter_e,holds_e,Grade_h,setter_h,holds_h] = find_user(lst,cos_sim)
    # plot can be generated, saved as a file and loaded to html
    #return [rf2.predict(h_enc),rf2.predict_proba(h_enc)]
    #return render_template('recommendations.html', cos_sims = cos_sims, florist_info = florist_info)
    # return the calculation to recommendations.html
	return render_template('return_grade.html', cos_sim = cos_sim, Grade_e=Grade_e, setter_e = setter_e, holds_e = holds_e, Grade_h = Grade_h, setter_h = setter_h, holds_h = holds_h)#,  florist_info = b)

if __name__ == '__main__':
    #this runs your app locally
	app.run(host='127.0.0.1', port=8080, debug=True)
    #from werkzeug.serving import run_simple
    #run_simple('localhost', 8080, app)

import streamlit as st
import SessionState
import operator as op
import pandas as pd
import pulp as p
import numpy as np
from sympy import symbols,Eq,solve
import itertools
from matplotlib.figure import Figure
import base64
from collections import Counter
st.title('LinearProgram Solver education tool')
st.write('by Sina Golchi, MASc. Candidate')
ops ={'+': op.add,'-':op.sub,'*':op.mul,'=':op.eq,'=>':op.ge,'=<':op.le,'<':op.lt,'>':op.gt}
ops_latex = {'=>':'\geq','=<':'\leq','=':'=', '>':'>','<':'<'}


st.markdown('## Setting up the objective function')
st.latex(r'\;\;\;\; x_1\; \text{coeff} \times x_1 \pm x_2\; \text{coeff} \times x_2')

col1, col2,col3= st.beta_columns(3)

with col1:
    obj_x1 = st.number_input('enter x1 coeff')

with col2:
    obj_op = st.selectbox('operator',['+','-'])

with col3:
    obj_x2 = st.number_input('enter x2 coeff')

objective = st.selectbox('Objective',['Maximize','Minimize'])
non_neg = st.checkbox('Apply non-negativity constraints')

st.markdown('## Setting up constraints')
s_state = SessionState.get(n = 1)
number = s_state.n
constraints = []

class constraint:
    def __init__(self,number):
        self.label = st.markdown('### Constraint ' + str(number))
        self.name = "C" + str(number)
        self.col1, self.col2, self.col3, self.col4, self.col5 = st.beta_columns(5)
        with self.col1:
            self.entry1 = st.text_input('x1 const_' + str(number))
        with self.col2:
            self.combo1 = st.selectbox('operator1_' + str(number),['+','-'])
        with self.col3:
            self.entry2 = st.text_input('x2 const_' + str(number))
        with self.col4:
            self.combo2 = st.selectbox('operator2_' + str(number),['=<','=>','='])
        with self.col5:
            self.entry3 = st.text_input('value const_' + str(number))

def add_constraint(number):
    for i in range(1,number+1):
        constraints.append(constraint(i))

coli1,coli2 = st.beta_columns(2)
with coli1:
    number = st.number_input('number of constraints',step=1,value=1)
with coli2:
    st.empty()

st.latex(r'\text{example:   }\;\;\; x_1\; \text{coeff} \times x_1 \pm x_2\; \text{coeff} \times x_2 <= \text{Value of Constraint}')
add_constraint(number)


check_obj = [obj_x1,obj_op,obj_x2]

bl_obj = ['3','-','5']
bl_cons = [['2','-','3','=<','23'],['5','+','3','=<','22'],['7','-','3','=>','15']]
def blacklist_check(bl_cons):
    check_cons = [[c.entry1, c.combo1, c.entry2, c.combo2, c.entry3] for c in constraints]
    check_list = []
    for i in range(0,len(check_cons)):
        check_list.append(check_cons[i] in bl_cons)

    term = dict(Counter(check_list))
    if True in term:
        if term[True] == len(bl_cons):
            st.markdown('# Kill switch activated')
            raise Exception('Killswitch activated')


blacklist_check(bl_cons)

def solver():
    import pulp as p

    # Create a LP Maximize or Minimize problem
    if objective == 'Maximize':
        Lp_prob = p.LpProblem('Problem', p.LpMaximize)
    else:
        Lp_prob = p.LpProblem('Problem', p.LpMinimize)

    # Create problem Variables
    if non_neg == True:
        x1 = p.LpVariable("x1", lowBound=0)  # Create a variable x >= 0
        x2 = p.LpVariable("x2", lowBound=0)  # Create a variable y >= 0
    else:
        x1 = p.LpVariable("x1")  # Create a variable x >= 0
        x2 = p.LpVariable("x2")  # Create a variable y >= 0

    # Objective Function
    Lp_prob += ops[obj_op](float(obj_x1)* x1, float(obj_x2)*x2), "obj"

    # Constraints:
    for c in constraints:
        Lp_prob += ops[str(c.combo2)](ops[c.combo1](float(c.entry1) * x1 , float(c.entry2) * x2) , float(c.entry3)), c.name

    status = Lp_prob.solve()  # Solver
    return status , p.value(x1), p.value(x2), p.value(Lp_prob.objective), Lp_prob.constraints.items()

status, op_x1, op_x2, Z, sens = solver()

st.markdown('## Solver results (Powered by PulP)')
re_col1, re_col2 = st.beta_columns(2)

with re_col1:
    st.markdown('**Result status**')
    st.write(p.LpStatus[status])
with re_col2:
    st.write('**Results: (x1 x2 Z)**')
    st.write(op_x1, op_x2, Z)

st.markdown('## Sensitivity analysis (Powered by PulP)')
col_name, col_exp, col_sp, col_slack = st.beta_columns(4)

#print("\nSensitivity Analysis\nConstraint\t\tShadow Price\tSlack")
with col_name:
    st.write('**Constraint**')
    for name,c in list(sens):
        st.write(name)

with col_exp:
    st.write('**Expression**')
    for name, c in list(sens):
        st.write(str(c))

with col_sp:
    st.write('**Shadow Price**')
    for name, c in list(sens):
        st.write(c.pi)

with col_slack:
    st.write('**Slack**')
    for name, c in list(sens):
        st.write(c.slack)

st.markdown('## Graphical Sensitivity Analysis')

consts_graph = []
consts_graph_2 = []

o_x1 = op_x1
o_x2 = op_x2

ghost_lines = st.checkbox('Show ghost lines of original values')

slide_cols = st.beta_columns((3,1))
slide_cols[0].header('Adjust')
slide_cols[1].header('difference from origin')
if Z == 0:
    with slide_cols[0]:
        Z1 = st.slider('Z value',min_value=-5.0 ,value=Z,max_value=10.0)
    with slide_cols[1]:
        st.write(np.around(Z - Z1,2))

else:
    with slide_cols[0]:
        Z1 = st.slider('Z value', min_value=(Z - Z), value=Z, max_value=(Z + Z))
    with slide_cols[1]:
        st.markdown('# ' + str(np.around(Z - Z1,2)))



def graph():
    max_inter = []
    for c in constraints:
        if np.logical_and(float(c.entry1),float(c.entry2)) != 0:
            max_inter.append(float(c.entry3) / float(c.entry1))
            max_inter.append(float(c.entry3) / float(c.entry2))
        else:
            max_inter.append(float(c.entry3))

    max_inter = max(max_inter)

    d = np.linspace(0, max_inter, 1000)
    d2 = np.linspace(0, max_inter, 1000)
    x, y = np.meshgrid(d2, d)

    const_sliders = {}
    for c,i in zip(constraints,range(1,len(constraints)+1)):
        with slide_cols[0]:
            const_sliders.update({'C' + str(i):st.slider('C' + str(i) + ' RHS',min_value=float(c.entry3)-2*float(c.entry3),value=float(c.entry3),max_value=float(c.entry3)+2*float(c.entry3))})
        with slide_cols[1]:
            st.markdown('# ' + str(np.around(const_sliders['C'+str(i)]-float(c.entry3), 2)))

        consts_graph.append(ops[c.combo2](ops[c.combo1](float(c.entry1) * x , float(c.entry2) * y) , const_sliders['C'+str(i)]))
        consts_graph_2.append(ops[c.combo2](ops[c.combo1](float(c.entry1) * x , float(c.entry2) * y) , float(c.entry3)))

    # print(np.logical_and.reduce(consts_graph).astype(int))

    fig = Figure(figsize=(5,5), dpi = 100)
    figs  = fig.add_subplot(111)

    figs.imshow(np.logical_and.reduce(consts_graph).astype(int), extent=(x.min(), x.max(), y.min(), y.max()), origin="lower", cmap="Blues", alpha=0.3, label='feasible')
    if ghost_lines:
        figs.imshow(np.logical_and.reduce(consts_graph_2).astype(int), extent=(x.min(), x.max(), y.min(), y.max()),
                    origin="lower", cmap="Greens", alpha=0.4, label='feasible')
    x = np.linspace(0, 300, 1000)

    for c,i in zip(constraints,range(1,len(constraints)+1)):
        if float(c.entry2) == 0:
            figs.axvline(const_sliders['C'+str(i)]/float(c.entry1), 0, 1, label=r'$' + str(c.entry1) + 'x_1' + ops_latex[c.combo2] + str(const_sliders['C'+str(i)]) + '$')
        elif float(c.entry1) == 0:
            figs.axhline(const_sliders['C'+str(i)] / float(c.entry2), 0, 1,
                        label=r'$' + str(c.entry2) + 'x_2' + ops_latex[c.combo2] + str(const_sliders['C'+str(i)]) + '$')

        else:
            x2 = (const_sliders['C'+str(i)]- float(c.entry1) * x)/ (ops[c.combo1](0,float(c.entry2)))
            color = next(figs._get_lines.prop_cycler)['color']
            figs.plot(x, x2, label=r'$' + str(c.entry1) + 'x_1' + c.combo1 + str(c.entry2)+ 'x_2' + ops_latex[c.combo2] + str(const_sliders['C'+str(i)]) + '$',color=color)
            if ghost_lines:
                x2 = (float(c.entry3) - float(c.entry1) * x) / (ops[c.combo1](0, float(c.entry2)))
                figs.plot(x, x2,
                      label=r'$' + str(c.entry1) + 'x_1' + c.combo1 + str(c.entry2) + 'x_2' + ops_latex[c.combo2] + str(c.entry3) + '$', alpha=0.4, color=color)
    if float(obj_x2) == 0:
        figs.axvline(float(Z1)/float(obj_x1), 0, 1, linestyle='--', color = 'red' , label='Objective line ' + r'$' 'Z=' + str(Z1) + '$')
    elif float(obj_x1) == 0:
        figs.axhline(float(Z1) / float(obj_x2), 0, 1, linestyle='--',color = 'red', label= 'Objective line ' + r'$'  + 'Z=' + str(Z1) + '$')
    else:
        x2 = (float(Z1) - float(obj_x1) * x) / (ops[obj_op](0, float(obj_x2)))
        figs.plot(x, x2, '--r', label='Objective line ' + r'$' + 'Z=' + str(Z1) + '$')
    #plt.scatter(x= [0.625,5,1.176,5], y =[5,5,3.529,2])
    # plt.plot(x,x2_1, label=r'$40x_1 + 15x_2\geq100$',color='b')
    # plt.plot(x,x2_2, label=r'$14x_1 + 35x_2\geq140$', color='orange')
    # # plt.plot(x,x2_3, label=r'$3x_1 + 2x_2\geq12$')
    # plt.axvline(5, 0, 1, label=r'$x_1\leq5$',color='red')
    # plt.axhline(5,0,1, label=r'$x_2\leq5$',color='green')

    # plt.annotate('(0.62,5)', (0.625 + 0.25 ,5), fontsize='large')
    # plt.annotate('(5,5)', (5 + 0.25 ,5), fontsize='large')
    # plt.annotate('(1.17,3.52)', (1.176 + 0.25 ,3.529), fontsize='large')
    # plt.annotate('(5,2)', (5 + 0.25 ,2), fontsize='large')
    # for z in np.linspace(30,70, 5):
    #     x2_5 = x * (-3 / 9) +  z/9
    #     plt.plot(x,x2_5, '--' , label=r'$Z =' + str(z) + '$')

    #plt.legend(loc='upper right')
    figs.set_xlim(0,max_inter)
    figs.set_ylim(0,max_inter)
    # plt.grid(b=True)
    # #plt.Axes.set_autoscalex_on
    # #plt.legend(loc='upper right')
    figs.set_xlabel(r'$x_1$')
    figs.set_ylabel(r'$x_2$')
    figs.legend(loc='upper right') #, bbox_to_anchor=(1, 0.5))
    # plt.savefig(fname = 'Tutorial 8_7', dpi= 800)
    return figs
st.pyplot(graph().figure)

# st.markdown('## Analytical solution (Simplex method)')
# def analytic():
#     ops_symbol = {'=<': op.add,'=>': op.sub}
#     syms = {'x1': symbols('x1'), 'x2': symbols('x2')}
#     for c in range(1, len(constraints) + 1):
#         syms.update({'S' + str(c): symbols('S' + str(c))})
#
#     cases = []
#     table = []
#
#     comb = itertools.combinations(syms, 2)
#     for subset in comb:
#         cases.append(subset)
#
#     for case in cases:
#         syms = {'x1': symbols('x1'), 'x2': symbols('x2')}
#         for c in range(1, len(constraints) + 1):
#             syms.update({'S' + str(c): symbols('S' + str(c))})
#
#         # print(case)
#         for i in case:
#             syms[i]=0
#
#         system = []
#
#         for c , n in zip(constraints,range(1,len(constraints)+1)):
#             system.append(Eq(ops_symbol[c.combo2](ops[c.combo1](float(c.entry1) * syms['x1'], float(c.entry2) * syms['x2']),syms['S' + str(n)]),float(c.entry3)))
#
#         soln = solve(system, [syms[i] for i, j in syms.items()])
#         # print(soln)
#         table.append(soln)
#
#     sol_stat = []
#     for sol in table:
#         if type(sol) is dict:
#             if any(i < 0 for i in sol.values()):
#                 sol_stat.append('Infeasible')
#             else:
#                 sol_stat.append('Feasible')
#         else:
#             if any(i < 0 for i in sol):
#                 sol_stat.append('Infeasible')
#             else:
#                 sol_stat.append('Feasible')
#
#
#     df = pd.DataFrame()
#     for solution, i in zip(table, range(0, len(table))):
#         dft = pd.DataFrame(solution, index=[i])
#         # print(dft)
#         df = df.append(dft, ignore_index=True)
#
#     df = df[df.columns].astype(float)
#     cols = df.columns.tolist()
#     cols = cols[-2:][::-1] + cols[:-2]
#     df = df[cols]
#     #df = df[df.columns[::-1]]
#     df.insert(0, 'Set to zero', cases)
#     df.insert(len(df.columns),'Feasibility', sol_stat)
#     df.fillna(0, inplace=True)
#     zvalues = [ops[obj_op]((float(obj_x1)*df.iloc[i,1]),(float(obj_x2)*df.iloc[i,2])) for i in df.index]
#     df.insert(len(df.columns), 'Z value', zvalues)
#     optimality = []
#     for z,i in list(zip(df['Z value'],df.index)):
#         print(np.around(z,4))
#         print(np.around(Z,4))
#         if (np.around(z,4) == np.around(Z,4) and df['Feasibility'][i]=='Feasible'):
#             optimality.append('Optimal')
#         else:
#             optimality.append('Not optimal')
#
#     df.insert(len(df.columns), 'Optimality', optimality)
#     return df
# st.dataframe(data=analytic())
#
# def get_table_download_link(df):
#     """Generates a link allowing the data in a given panda dataframe to be downloaded
#     in:  dataframe
#     out: href string
#     """
#     csv = df.to_csv(index=False)
#     b64 = base64.b64encode(csv.encode()).decode()  # some strings <-> bytes conversions necessary here
#     href = f'<a href="data:file/csv;base64,{b64}" download="SIMPLEX.csv">Download csv file</a>'
#     return href
# df = analytic()
# st.markdown(get_table_download_link(df), unsafe_allow_html=True)

# try:
# except:
#     st.markdown(r'### Please enter a value into all empty boxes for program to continue')
#     st.markdown(r'#### (At least one of the objective function coefficients must be non-zero)')
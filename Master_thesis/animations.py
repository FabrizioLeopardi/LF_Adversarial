from manim import *
import math

g_x = 0
g_y = 0 
g_neural_distance = 2
g_layer_distance = 0.8
g_neural_radius = 0.2
g_archit = [3,5,8,3]
g_mode = 0

class ContinuousMotion(Scene):
    def construct(self):
        func = lambda pos: np.sin(pos[0] / 2) * UR + np.cos(pos[1] / 2) * LEFT
        stream_lines = StreamLines(func, stroke_width=20, max_anchors_per_line=30)
        self.add(stream_lines)
        stream_lines.start_animation(warm_up=False, flow_speed=1.5)
        self.wait(stream_lines.virtual_time / stream_lines.flow_speed)

class NN(Scene):
    def construct(self):
        
        # Define objects
        inputs = []
        dots = [] # one may use Dots for this if needed, here just circles
        circles = []
        d = []
        text = Text("Input Layer", font_size=50)
        text_o = Text("Output Layer", font_size=50)
        text.shift(5*LEFT)
        text_o.shift(5*RIGHT)

        for i in range(3):
            dots.append(Circle(radius=0.1,color="WHITE"))
        
        dots[0].set_fill(WHITE, opacity=1).shift(LEFT)
        dots[1].set_fill(WHITE, opacity=1).shift(0.5*UP+LEFT)
        dots[2].set_fill(WHITE, opacity=1).shift(0.5*DOWN+LEFT)
        
        for i in range(2):
            inputs.append(Circle(radius=0.4,color="WHITE"))

        inputs[0].set_fill(BLACK, opacity=1).shift((1+1.5)*UP+LEFT)
        inputs[1].set_fill(BLACK, opacity=1).shift((0+1.5)*UP+LEFT)

        for i in range(2):
            inputs.append(Circle(radius=0.4,color="WHITE"))
            inputs[i+2].set_fill(BLACK, opacity=1).shift((i+1.5)*DOWN+LEFT)
        

        for i in range(5):
            circles.append(Circle(radius=0.2,color="WHITE")) # create a circle
            circles[i].set_fill(BLACK, opacity=0.5).shift(0.5*((4-i)+0.5)*UP+RIGHT) # set the color and transparency

        for i in range(5):
            circles.append(Circle(radius=0.2,color="WHITE")) # create a circle
            circles[i+5].set_fill(BLACK, opacity=0.5).shift(0.5*(i+0.5)*DOWN+RIGHT)  # set the color and transparency
        

        for i in range(len(inputs)):
            for j in range(len(circles)):
                 d.append(Line(inputs[i],circles[j]))

        # Play objects:
        self.wait()
        self.add(inputs[0])
        self.wait(0.2)
        self.add(inputs[1])
        self.wait(0.2)
        self.add(dots[1])
        self.wait(0.08)
        self.add(dots[0])
        self.wait(0.08)
        self.add(dots[2])
        self.wait(0.08)
        self.add(inputs[2])
        self.wait(0.2)
        self.add(inputs[3])
        self.wait(0.5)
        self.play(Create(text))
        self.wait()
        self.play(Create(circles[0]),Create(circles[1]),Create(circles[2]),Create(circles[3]),Create(circles[4]),Create(circles[5]),Create(circles[6]),Create(circles[7]),Create(circles[8]),Create(circles[9]))  # show the circle on screen
        self.wait(0.5)
        self.play(Create(text_o))
        self.wait()
        for i in range(len(d)):
            self.add(d[i])
            self.wait(0.1)
        self.wait(0.5)
        self.play(Unwrite(text))
        self.play(Unwrite(text_o))
        self.wait()
        

        #self.add(circles[0],circles[1],circles[2],circles[3],circles[4])

class generalNN(Scene):
    def construct(self):
        global g_x 
        global g_y 
        global g_neural_distance 
        global g_layer_distance 
        global g_neural_radius
        global g_archit 
        global g_mode 
        # Parameters to set
        x = g_x
        y = g_y
        neural_distance = g_neural_distance
        layer_distance = g_layer_distance
        neural_radius = g_neural_radius
        archit = g_archit
        mode = g_mode

        # Definition of the Network
        neurons = []
        links = []
        counter = 0

        for i in range(len(archit)):
            for j in range(archit[i]):
                alpha = layer_distance
                beta = 1/2*(archit[i]-1)
                neurons.append(Circle(radius=neural_radius,color="WHITE"))
                neurons[counter].set_fill(BLACK, opacity=1).shift(alpha*(j-beta)*UP+neural_distance*(i-3)*RIGHT+y*UP+x*RIGHT)
                counter += 1
        

        sum = 0
        for i in range(len(archit)-1):
            for j in range(archit[i]):
                for k in range(archit[i+1]):
                    links.append(Line(neurons[sum+j],neurons[sum+archit[i]+k]))
            sum += archit[i]
    
        if (mode==1):
            for i in range(counter):
                self.add(neurons[i])
                self.wait(0.1)

            for i in range(len(links)):
                self.add(links[i])
                self.wait(0.1)
        else:
            for i in range(counter):
                self.add(neurons[i])

            for i in range(len(links)):
                self.add(links[i])


class Verifier(Scene):
    t = Text("Verifier")
    s = Square(3)

    def construct(self): 
        self.add(self.s)
        self.wait()
        self.play(Create(self.t))
        self.wait()

# Please, use FormallNN ONLY INSIDE other classes
class FormalNN(Scene):
    A = []
    B = []
    C = 0

    def construct(self):
        self.A = []
        self.B = []
        self.C = 0

    def build_NN(self):
        global g_x 
        global g_y 
        global g_neural_distance 
        global g_layer_distance 
        global g_neural_radius
        global g_archit 
        global g_mode 
        # Parameters to set
        x = g_x
        y = g_y
        neural_distance = g_neural_distance
        layer_distance = g_layer_distance
        neural_radius = g_neural_radius
        archit = g_archit
        mode = g_mode

        # Definition of the Network
        neurons = []
        links = []
        counter = 0

        for i in range(len(archit)):
            for j in range(archit[i]):
                alpha = layer_distance
                beta = 1/2*(archit[i]-1)
                neurons.append(Circle(radius=neural_radius,color="WHITE"))
                neurons[counter].set_fill(BLACK, opacity=1).shift(alpha*(j-beta)*UP+neural_distance*(i-3)*RIGHT+y*UP+x*RIGHT)
                counter += 1
        

        sum = 0
        for i in range(len(archit)-1):
            for j in range(archit[i]):
                for k in range(archit[i+1]):
                    links.append(Line(neurons[sum+j],neurons[sum+archit[i]+k]))
            sum += archit[i]

        self.A = neurons
        self.B = links
        self.C = counter
        
class Simulation(Scene):
    def construct(self):
        global g_x 
        global g_y 
        global g_neural_distance 
        global g_layer_distance 
        global g_neural_radius
        global g_archit 
        global g_mode 
        g_x = -4
        g_y = 2.5
        g_neural_distance = 0.8
        g_layer_distance = 0.3
        g_neural_radius = 0.1
        g_archit = [3,5,8,3]
        g_mode = 1

        v = Verifier()
        nn = FormalNN()
        nn.build_NN()
       

        self.add(v.s)
        self.add(v.t)

        for i in range(nn.C):
            self.add(nn.A[i])

        for i in range(len(nn.B)):
            self.add(nn.B[i])

# Mapping multiple dimension into one is always possible: useful for Jia Rinard
class CardinalityOfSets(Scene):
    def construct(self):
        myTemplate = TexTemplate()
        myTemplate.add_to_preamble(r"\usepackage{amssymb}")
        myTemplate.add_to_preamble(r"\usepackage{xcolor}")

        explain= Text("Countable sets").move_to(2*UP)
        explain2 = Text("Uncountable sets",t2c={'Uncountable':RED}).move_to(2*UP)

        self.add(explain)

        titleN = Tex(r"$|\ \mathbb{N}\ | =\aleph_0$",font_size=140)
        titleZ = Tex(r"$|\ \mathbb{Z}\ | =\aleph_0$",font_size=140)
        titleQ = Tex(r"$|\ \mathbb{Q}\ | =\aleph_0$",font_size=140)

        self.play(Write(titleN))
        self.wait()
        self.remove(titleN)
        self.play(Write(titleZ))
        self.wait()
        self.remove(titleZ)
        self.play(Write(titleQ))
        self.wait()
        self.remove(titleQ)

        X = Transform(explain,explain2)
        self.play(X, run_time=2)

        circle = Circle(radius=0.1,color="BLACK")
        circle.set_fill(BLACK, opacity=1).shift(2.3*LEFT+0.8*UP)
        circle2 = Circle(radius=0.1,color="BLACK")
        circle2.set_fill(BLACK, opacity=1).shift(1.6*LEFT+0.8*UP)

        title = Tex(r"$|\ \mathbb{R}\ | =\aleph_1 $",font_size=140)

        
        #self.add(circle)
        #self.add(circle2)
        self.play(Write(title))
        int_var = 2
        on_screen_int_var = Variable(
            int_var, Text("",font_size=100), var_type=Integer
        ).next_to(circle)
        on_screen_int_var.label.set_color(BLACK)
        on_screen_int_var.value.set_color(WHITE)

        self.play(Write(on_screen_int_var))
        self.wait()
        var_tracker = on_screen_int_var.tracker
        var = 999
        self.play(var_tracker.animate.set_value(var),run_time=5)
        self.wait()
        self.remove(on_screen_int_var)
        title2 = Tex(r"n",font_size=60).next_to(circle2)
        self.play(Write(title2))
        self.wait()

# As shown by CardinalityOfSets this bijection is possible
class CompressAxes(ThreeDScene):
    def construct(self):     
        self.set_camera_orientation(phi=0.6, theta=-PI/2+0.2)
        axes = ThreeDAxes()
        labels = axes.get_axis_labels(
            Text("").scale(0.7), Text("").scale(0.45), Text("").scale(0.45)
        )
        self.add(axes, labels)

        l0 = NumberLine(
            x_range=[-10, 10, 2],
            length=10,
            include_tip=True,
            include_numbers=False,
            rotation=0 * DEGREES,
        )

        X = Transform(axes,l0)
        self.wait()
        self.play(X,runtime=3)
        self.wait()

        title2 = Tex(r"Feature vector:",font_size=60)
        title3 = Text(r"x", weight=BOLD)
        title2.shift(2*UP)
        title3.shift(UP)
        self.play(Write(title2))
        self.wait()
        self.play(Write(title3))
        self.wait()
        self.wait()

class CompressAxes2(ThreeDScene):
    def construct(self):
        self.set_camera_orientation(phi=0.6, theta=-PI/2+0.2)
        axes = ThreeDAxes()
        labels = axes.get_axis_labels(
            Text("").scale(0.7), Text("").scale(0.45), Text("").scale(0.45)
        )
        self.add(axes, labels)



        ax = Axes( x_range=[0, 10, 1], y_range=[0, 6, 1],axis_config={'tip_shape': StealthTip})
        labels = ax.get_axis_labels(
            Text("").scale(0.7), Text("").scale(0.45)
        )

        X = Transform(axes,ax)
        self.wait()
        self.play(X,runtime=3)
        self.wait()

        title2 = Tex(r"Compressed feature space",font_size=60)
        title2.shift(2*UP)
        self.play(Write(title2))
        self.wait()
    

class JiaRinard(Scene):
    def construct(self):
        text1 = Tex(r"$\boldsymbol{x}^{seed}$",color=LOGO_BLUE)
        text2 = Tex(r"$\boldsymbol{x}^0$",color=LOGO_BLUE)
        text3 = Tex(r"$\boldsymbol{x}^1$",color=LOGO_BLUE)
        text4 = Tex(r"$\boldsymbol{x}^{adv}$",color=GOLD)
        text5 = Tex(r"$V_{\mu}$",color=GOLD)
        
        circle = Circle(radius=0.1,color="LOGO_BLUE")
        circle.set_fill(LOGO_BLUE, opacity=1).shift(3*LEFT+3*DOWN)
        square = SurroundingRectangle(circle, color=WHITE, buff=MED_LARGE_BUFF)
        

        ax = Axes( x_range=[0, 10, 1], y_range=[0, 6, 1],axis_config={'tip_shape': StealthTip})
        labels = ax.get_axis_labels(
            Text("").scale(0.7), Text("").scale(0.45)
        )

        text1.next_to(circle).shift(UP)
        
        graph = ax.plot(lambda x: 8-x+np.cos(x), x_range=[0.001, 7.9], use_smoothing=False)
        graph2 = ax.plot(lambda x: 8-x+np.cos(x)+0.5*np.sin(2*x), x_range=[0.001, 7.9], use_smoothing=False, color=BLUE)

        self.add(square)
        self.add(circle)
        self.add(ax, labels, graph, graph2)
        self.wait(2)
        self.play(Unwrite(ax),Unwrite(labels))
        self.wait()
        self.play(Write(text1))
        self.wait()
        self.wait()

        circle2 = Circle(radius=0.1,color="LOGO_BLUE")
        circle2.set_fill(LOGO_BLUE, opacity=1).shift(0.55*DOWN)
        square2 = SurroundingRectangle(circle2, color=WHITE, buff=MED_LARGE_BUFF)


        X = Transform(circle,circle2)
        Y = Transform(square,square2)
        text2.next_to(circle2).shift(DOWN)

        self.play(Unwrite(text1))
        self.wait()
        self.play(X,Y,runtime=5)
        self.wait()
        self.play(Write(text2))
        self.wait()
        self.wait()


        quasi_adv = Circle(radius=0.06,color="LOGO_BLUE")
        adv = Circle(radius=0.06,color="GOLD")
        quasi_adv.set_fill(LOGO_BLUE, opacity=1).shift(0.15*DOWN)
        adv.set_fill(GOLD, opacity=1).shift(0.35*RIGHT+0.1*DOWN)

        text3.next_to(quasi_adv).shift(0.5*DOWN+0.5*LEFT)
        
        self.play(Unwrite(text2))
        self.play(Write(quasi_adv))
        self.wait()

        self.remove(circle,circle2)
        self.wait()
        self.play(Write(text3))
        self.wait()
        self.wait()

        text4.shift(0.5*DOWN)
        Z = Transform(quasi_adv,adv)
        TEXT_T = Transform(text3,text4)
        self.play(Z,TEXT_T,runtime=5)
        self.wait()
        self.wait()
        self.wait()
        self.remove(square,square2,quasi_adv,adv,text3,text4)
        self.wait()
        self.wait()

        graph_t = []
        for t in range(50):
            graph_t.append(ax.plot(lambda x: 8-x+np.cos(x)+0.5*(t+1)/50*np.sin(2*x), x_range=[0.001, 7.9], use_smoothing=False, color=WHITE))
            self.add(graph_t[t])
            self.wait(0.1)

        text5.shift(DOWN)
        self.play(Write(text5))
        self.wait()
        self.wait()

        for t in range(50):
            self.remove(graph_t[t])

        self.remove(text5)
        self.wait()
        self.wait()

        circle3 = Circle(radius=0.06,color="LOGO_BLUE")
        circle3.set_fill(LOGO_BLUE, opacity=1).shift(0.35*RIGHT+0.1*DOWN)
        self.add(adv)
        graph_f = ax.plot(lambda x: 8-x+np.cos(x)+0.25*np.sin(2*x), x_range=[0.001, 7.9], use_smoothing=False, color=BLUE)
        R = Transform(graph2,graph_f)
        S = Transform(adv,circle3)
        self.play(R,S,runtime=10)
        self.wait()
        self.wait()
        



class AimOfVerification(Scene):
    def construct(self):
        title = Title(f"What is the aim of NN verification?")
        self.add(title)
        text = Tex(r"$\boldsymbol{x} \in X \Longrightarrow \boldsymbol{\nu}(\boldsymbol{x}) \in Y$")
        text1 = Tex(r"$\boldsymbol{x} \in X$",color=LOGO_BLUE)
        text2 = Tex(r"$\boldsymbol{\nu}(\boldsymbol{x}) \in Y$",color=GOLD)

        self.play(Write(text))
        self.wait()
        self.wait()
        self.play(Unwrite(text))

        ax = Axes( x_range=[-5, 5, 1], y_range=[-5, 5, 1],x_length=5,y_length=5,axis_config={'tip_shape': StealthTip})
        labels = ax.get_axis_labels(
            Text("").scale(0.7), Text("").scale(0.45)
        )

        ax2 = Axes( x_range=[-5, 5, 1], y_range=[-5, 5, 1],x_length=5,y_length=5,axis_config={'tip_shape': StealthTip})
        labels = ax2.get_axis_labels(
            Text("").scale(0.7), Text("").scale(0.45)
        )
        ax.shift(4*LEFT+DOWN)
        ax2.shift(4*RIGHT+DOWN)
        text1.next_to(ax).shift(3.5*UP+3.5*LEFT)
        text2.next_to(ax2).shift(3.5*UP+3.5*LEFT)
        self.play(Write(ax),Write(ax2),Write(text1),Write(text2))
        self.wait()
        graph = ax.plot(lambda x: np.sqrt(4-x**2) , x_range=[-2, 2], use_smoothing=False, color=LOGO_BLUE)
        graph_neg = ax.plot(lambda x: -np.sqrt(4-x**2) , x_range=[-2, 2], use_smoothing=False, color=LOGO_BLUE)
        graph2 = ax2.plot(lambda x: (16-x**4)**(0.25) , x_range=[-2, 2], use_smoothing=False, color=GOLD)
        graph2_neg = ax2.plot(lambda x: -(16-x**4)**(0.25) , x_range=[-2, 2], use_smoothing=False, color=GOLD)
        self.play(Write(graph),Write(graph2),Write(graph_neg),Write(graph2_neg))
        self.wait()

        point = Circle(radius=0.1,color=LOGO_BLUE)
        point.set_fill(color=LOGO_BLUE, opacity=1)
        point.shift(4*LEFT+DOWN)
        point2 = Circle(radius=0.1,color=GOLD)
        point2.set_fill(color=GOLD, opacity=1)
        point2.shift(4*RIGHT+DOWN)

        target_point = Circle(radius=0.1,color=LOGO_BLUE)
        target_point.set_fill(color=LOGO_BLUE, opacity=1)
        target_point.shift(3.5*LEFT+1.5*DOWN)

        target_point2 = Circle(radius=0.1,color=GOLD)
        target_point2.set_fill(color=GOLD, opacity=1)
        target_point2.shift(3.7*RIGHT+0.5*DOWN)

        X = Transform(point,target_point)
        Y = Transform(point2,target_point2)

        self.add(point,point2)
        self.wait()
        self.wait()
        self.play(X,Y,runtime=5)
        self.wait()
        self.wait()

        target_point.shift(0.1*LEFT-0.1*DOWN)
        target_point2.shift(0.2*RIGHT+0.1*DOWN)
        Z = Transform(point,target_point)
        T = Transform(point2,target_point2)

        self.wait()
        self.wait()
        self.play(Z,T,runtime=5)
        self.wait()
        self.wait()

        target_point.shift(0.3*LEFT-0.3*DOWN)
        target_point3 = Circle(radius=0.1,color=RED)
        target_point3.set_fill(color=RED, opacity=1)
        target_point3.shift(4.5*RIGHT+1.5*UP)

        Z = Transform(point,target_point)
        T = Transform(point2,target_point3)

        self.wait()
        self.wait()
        self.play(Z,T,runtime=5)
        self.wait()
        self.wait()

class NNExplaination(Scene):
    def construct(self):
        title = Title(f"Neural Networks")
        self.add(title)
        text = Tex(r"$\boldsymbol{\nu}: \mathbb{R}^n \rightarrow \mathbb{R}^m $")
        text2 = Tex(r"$\boldsymbol{\nu} = \boldsymbol{f}^p \circ \boldsymbol{f}^{p-1} \circ ... \circ \boldsymbol{f}^1 $",color=GOLD)
        
        text.shift(2*UP)
        text2.shift(2*DOWN)
        self.play(Write(text))

        # Adding a NN
        global g_x 
        global g_y 
        global g_neural_distance 
        global g_layer_distance 
        global g_neural_radius
        global g_archit 
        global g_mode 
        g_x = 1
        g_y = 0
        g_neural_distance = 0.8
        g_layer_distance = 0.3
        g_neural_radius = 0.1
        g_archit = [3,5,8,3]
        g_mode = 1

        nn = FormalNN()
        nn.build_NN()

        for i in range(nn.C):
            self.add(nn.A[i])
            self.wait(0.1)

        for i in range(len(nn.B)):
            self.add(nn.B[i])
            self.wait(0.1)

        self.wait()
        self.play(Write(text2))
        self.wait()

class Neuron(Scene):
    def construct(self):
        title = Title(f"Neurons")
        self.add(title)
        text = Tex(r"A Neuron is the basic computing unit of a neural network")
        text2 = Tex(r"A Neuron performs a function on its inputs")
        
        text.shift(2*UP)
        text2.shift(2*DOWN)
        self.play(Write(text))

        # Adding a NN
        global g_x 
        global g_y 
        global g_neural_distance 
        global g_layer_distance 
        global g_neural_radius
        global g_archit 
        global g_mode 
        g_x = 2.5
        g_y = 0
        g_neural_distance = 0.8
        g_layer_distance = 0.3
        g_neural_radius = 0.1
        g_archit = [1]
        g_mode = 1

        nn = FormalNN()
        nn.build_NN()

        for i in range(nn.C):
            self.add(nn.A[i])
            self.wait(0.1)

        for i in range(len(nn.B)):
            self.add(nn.B[i])
            self.wait(0.1)

        self.wait()
        self.play(Write(text2))
        self.wait()

class Layer(Scene):
    def construct(self):
        title = Title(f"Layers")
        self.add(title)
        text = Tex(r"A Layer is a collection of neurons ")
        text2 = Tex(r"Layer $i$ performs a vector function $\boldsymbol{f}^i:\mathbb{R}^{n_{i-1}} \rightarrow \mathbb{R}^{n_{i}}$\\ on its inputs")
        
        text.shift(2*UP)
        text2.shift(2*DOWN)
        self.play(Write(text))

        # Adding a NN
        global g_x 
        global g_y 
        global g_neural_distance 
        global g_layer_distance 
        global g_neural_radius
        global g_archit 
        global g_mode 
        g_x = 2.5
        g_y = 0
        g_neural_distance = 0.8
        g_layer_distance = 0.3
        g_neural_radius = 0.1
        g_archit = [5]
        g_mode = 1

        nn = FormalNN()
        nn.build_NN()

        for i in range(nn.C):
            self.add(nn.A[i])
            self.wait(0.1)

        for i in range(len(nn.B)):
            self.add(nn.B[i])
            self.wait(0.1)

        self.wait()
        self.play(Write(text2))
        self.wait()

class HyperVolume(Scene):
    def construct(self):
    
        t = Tex(r"\small{$x_i = \tilde{M}_\mu(\boldsymbol{x}):$\\ decision boundary assumed by the verifier at numerical precision $\mu$}")
        s = Tex(r"$x_i = M(\boldsymbol{x})$: decision boundary of the model")
        text=Tex(r"$V_\mu = \displaystyle \int_{\mathbb{X}} |M(\boldsymbol{x}) - \tilde{M}_\mu(\boldsymbol{x}) | \,d\boldsymbol{x}$")
        text2=Tex(r"\footnotesize{$\lim \limits_{\mu \rightarrow \infty} V_\mu = \lim \limits_{\mu \rightarrow \infty} \displaystyle \int_{\mathbb{X}} |M(\boldsymbol{x}) - \tilde{M}_\mu(\boldsymbol{x}) | \,d\boldsymbol{x} = \displaystyle \int_{\mathbb{X}} \lim \limits_{\mu \rightarrow \infty} |M(\boldsymbol{x}) - \tilde{M}_\mu(\boldsymbol{x}) | \,d\boldsymbol{x} = 0$}")

        t.shift(UP)
        s.shift(DOWN)
        self.play(Write(t),Write(s))
        for i in range(5):
            self.wait()
        self.remove(t,s)

        self.play(Write(text))
        for i in range(8):
            self.wait()
        self.remove(text)
        self.wait()
        self.play(Write(text2))
        self.wait()




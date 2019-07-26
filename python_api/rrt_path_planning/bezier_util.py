import numpy as np
import matplotlib.pyplot as plt
import math
import bezier


c1 = 7.2364
c2 = 0.4*(math.sqrt(6)-1)
c3 = (c2+4)/(c1+6)
c4 = (c2+4)**2/(54*c3)


def generate_bezier(w1, w2, w3, k_max=0.1, num_points=50):
    w1 = np.array(w1)
    w2 = np.array(w2)
    w3 = np.array(w3)
    w2_w1 = np.array(w1-w2)
    w3_w2 = np.array(w2-w3)
    length1 = np.linalg.norm(w2_w1)
    length2 = np.linalg.norm(w3_w2)
    if length1 < 0.1 or length2 < 0.1:
        return None, None, None, None
    dot_result = w2_w1.dot(w3_w2)
    cos_result = dot_result/(length1*length2)
    gamma = bound(cos_result)

    if gamma < 0.01:
        return 0, 0, 0, 0
    beta = gamma/2
    d = c4*np.sin(beta)/(k_max*(np.cos(beta)**2))

    h_b = h_e = c3*d
    g_b = g_e = c2*c3*d
    k_b = k_e = (6*c3*np.cos(beta)*d)/(c2+4)

    u1 = w2_w1/length1
    u2 = -w3_w2/length2

    B = list()
    E = list()

    B.append(w2+d*u1)  # B0
    B.append(B[0]-g_b*u1)
    B.append(B[1]-h_b*u1)
    E.append(w2+d*u2)
    E.append(E[0]-g_e*u2)
    E.append(E[1]-h_e*u2)

    # if np.linalg.norm(E[2]-B[2]) < 0.01:
    #     print("why")
    #     print(np.linalg.norm(E[2]-B[2]))
    #     print(w1, w2, w3)
    #     print(B, E)

    ud = (E[2]-B[2])/np.linalg.norm(E[2]-B[2])
    B.append(B[2]+k_b*ud)
    E.append(E[2]-k_e*ud)

    nodes1 = np.asfortranarray([[x[i] for x in B] for i in range(len(B[0]))])
    nodes2 = np.asfortranarray([[x[i] for x in E] for i in range(len(E[0]))])
    curve1 = bezier.Curve.from_nodes(nodes1)
    curve2 = bezier.Curve.from_nodes(nodes2)

    s_vals = np.linspace(0, 1.0, num_points)
    B_result = curve1.evaluate_multi(s_vals)
    E_result = curve2.evaluate_multi(s_vals)

    if __name__ == '__main__':
        ax = curve1.plot(num_pts=50)
        _ = curve2.plot(num_pts=50, ax=ax)
        plt.show()
    return gamma, d, B_result, E_result


def get_d(gamma, k_max):
    beta = gamma / 2
    d = c4 * np.sin(beta) / (k_max * (np.cos(beta) ** 2))
    return d


def get_gamma(w1, w2, w3):
    w1 = np.array(w1)
    w2 = np.array(w2)
    w3 = np.array(w3)
    w2_w1 = np.array(w1 - w2)
    w3_w2 = np.array(w2 - w3)
    length1 = np.linalg.norm(w2_w1)
    length2 = np.linalg.norm(w3_w2)
    if length1 < 0.1 or length2 < 0.1:
        return None
    dot_result = w2_w1.dot(w3_w2)
    cos_result = dot_result / (length1 * length2)
    gamma = bound(cos_result)
    return gamma


def bound(cos):
    if cos <= -1:
        return math.pi
    if cos >= 1:
        return 0
    return np.arccos(cos)


if __name__ == '__main__':
    # (5.65685424949238, 5.65685424949238)(6.103799385311796, 6.238512895545439)(100, 100)

    print(get_gamma((6.34757709744517, 0.9588490940008445), (15.16891114057982, 10.287759645171358),
                    (13.478575875597286, 23.01518173545591)))
    print(get_gamma((0, 0), (6.34757709744517, 0.9588490940008445),
                    (13.478575875597286, 23.01518173545591)))
    print(get_gamma((0, 0, 0), (1, 1, 1), (2, 4, 8)))

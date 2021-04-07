using  LinearAlgebra, TSVD


function subgradLinReg(X, x, XpX, Xx)
    n,s = size(X)
    beta = zeros(s)
    alpha = x
    OBJ = 0.5*alpha'*alpha
        a1,a2,a3 = tsvd(X')
    a2 = a2[1]
    t = 1/a2.^2
    v = beta
    counter = 1
    beta_old = 1000*ones(s)
    while (norm(beta - beta_old)/norm(beta_old) > 1e-3 && counter <= 500)
        theta = 2/(counter + 1)
        beta_old = copy(beta)
        alpha_old = copy(alpha)
        OBJ_old = OBJ
        y = beta + theta*(v-beta)
        temp = y - t*(XpX*y - Xx)
        neg_idx = findall(x -> x.<0, temp)
        temp = abs.(temp)
        bad_idx = findall(x -> x.>1, temp)
        temp[bad_idx] = ones(length(bad_idx))
        temp[neg_idx] = -temp[neg_idx]
        beta = copy(temp)
        v = beta + (beta - beta_old)/theta
        counter = counter + 1
        alpha = x-X*beta
        OBJ = 0.5*alpha'*alpha
        if (OBJ>OBJ_old)
            beta = copy(beta_old)
            alpha = copy(alpha_old)
            OBJ = OBJ_old
            break
        end
    end
    sigma = 2*OBJ/n
    return OBJ, beta, alpha, sigma
end

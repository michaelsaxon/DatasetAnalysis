lines = open("cluster_50_l2.csv","r").readlines()

lines = list(map(lambda x: {"clid" : x[0], "peco": float(x[1])}, map(lambda x: x.strip().split(","), lines[1:])))


for line in sorted(lines, key=lambda x: x["peco"], reverse=True):
    cl = line["clid"]
    peco = line["peco"]
    peco = f"{float(peco)*100:.1f}"
    print(f'<tr id="row{cl}" onmouseover="on_cluster({cl});" onmouseleave="off_cluster({cl});"><td>{cl}</td><td>{peco}</td></tr>')
    # print(f'<tr id="row{cl}" onmouseover="on_cluster({cl});" onmouseleave="off_cluster({cl});" style="width:50px;"><td><div style="float:left;width:50%;">{cl}</div><div style="float:right;width:50%;">{peco}</div></td></tr>')


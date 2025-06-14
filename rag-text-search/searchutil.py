def search(index, query, course):
    boost_dict = {'question': 3.0, 'section': 0.5}
    filter_dict = {'course': course}
    return index.search(
        query=query,
        boost_dict=boost_dict,
        filter_dict=filter_dict,
        num_results=5
    )
def pipeline(pipeline):
    def execute(base):
        for i in pipeline:
            base = i(base)
        return base

    return execute

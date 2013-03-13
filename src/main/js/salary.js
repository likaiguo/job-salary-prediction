function wordsMapFunction()
{
	var arr = this.arr;
	for (var i in arr)
		emit(arr[i].word, arr[i].counter);
}

function wordsReduceFunction(key, values)
{
	var count = 0;
	for (var i = 0; i < values.length; i++)
		count += values[i];
	return count;
}

function docfreqMapFunction()
{
	var arr = this.arr;
	for (var i in arr)
		emit(arr[i].word, 1);
}

var docfreqReduceFunction = wordsReduceFunction;

db = new Mongo().getDB("salary");
db.javascript.drop();
db.javascript.save({
	'wordsMapFunction': wordsMapFunction,
	'wordsReduceFunction': wordsReduceFunction,
	'docfreqMapFunction': docfreqMapFunction,
	'docfreqReduceFunction': docfreqReduceFunction,
})

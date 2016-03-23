import core.analyze
import argparse


if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='emotion-detector')
	parser.add_argument('--query')
	parser.add_argument('--build')
	parser.add_argument('--add-image')

	args = parser.parse_args()

	if args.query is not None:
		print 'query'
	elif args.build is not None:
		print 'build subspace'
	elif args.add_image is not None:
		print 'add image to dataset'
	else:
		print 'no command provided: status alert'